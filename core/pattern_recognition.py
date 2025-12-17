from datetime import datetime, timedelta

from django.db.models import F, Q, Window, Min, Max, DecimalField
from django.db.models.functions import TruncWeek
from django.db.models.expressions import RawSQL
from django.db.models.functions import Extract, Lead

from marketdata.models import EodPrice, Parameter


BOWL_MIN_DURATION_DAYS = 60
BOWL_LOCAL_MIN_WINDOW_DAYS = 2
BOWL_LEFT_LOOKBACK_MIN_DAYS = 20
BOWL_LEFT_LOOKBACK_MAX_DAYS = 90
BOWL_RIGHT_LOOKAHEAD_MIN_DAYS = 20
BOWL_RIGHT_LOOKAHEAD_MAX_DAYS = 90
BOWL_MIN_DEPTH = 0.06
BOWL_RIM_TOLERANCE = 0.30
BOWL_MIN_TOTAL_DAYS = 40
BOWL_BREAKOUT_LOOKAHEAD_DAYS = 120

# Default NRB window if frontend doesn't send a valid weeks value
NRB_LOOKBACK = 7

# Default cooldown period in weeks
DEFAULT_COOLDOWN_WEEKS = 4


def get_pattern_triggers(
    scrip: str,
    pattern: str,
    nrb_lookback: int,
    success_rate: float,
    weeks: int = 20,
    series: str | None = None,
    cooldown_weeks: int = DEFAULT_COOLDOWN_WEEKS,
):
    """
    Main entry to compute pattern triggers for a given symbol and pattern type.
    Returns a list of dicts; each dict at minimum has:
      - time (unix ts in seconds)
      - score (float)

    For NRB, additional fields are attached:
      - direction: "Bullish Break"
      - range_low, range_high
      - range_start_time, range_end_time
      - nrb_id
      - total_nrb_duration_weeks (same for all NRBs - total span)

    For Bowl, additional:
      - pattern_id
      
    Parameters:
      - cooldown_weeks: Minimum number of weeks between NRB detections (default: 4)
    """
    series_normalized = (series or "").strip().lower()

    if pattern == "Narrow Range Break":

        # CASE 1: DEFAULT – use price candles (EodPrice)
        if series_normalized in ("", "price", "close", "closing_price"):
            base_queryset = EodPrice.objects.filter(symbol__symbol=scrip)
            weekly_qs = get_weekly_queryset(base_queryset)
            total_weeks = weekly_qs.count()

            nr_weeks = weeks if weeks and weeks > 0 else NRB_LOOKBACK

            if total_weeks < nr_weeks + 1:
                return []

            weekly_data = list(
                weekly_qs.values(
                    "date", "high", "low", "close", "week"
                )
            )

            if not weekly_data:
                return []

            # ROLLING-BASED detection with cooldown
            triggers = _detect_narrow_range_break_rolling(weekly_data, nr_weeks, cooldown_weeks)
            triggers = _attach_daily_breakout_times_price(base_queryset, triggers)
            
            # 🆕 Calculate total NRB duration and attach to all triggers
            triggers = _calculate_total_nrb_duration(triggers)

            return triggers

        # CASE 2: PARAMETER-BASED SERIES
        else:
            PARAM_FIELD_MAP = {
                "ema21": "ema21",
                "ema50": "ema50",
                "ema200": "ema200",
                "rsc30": "rsc_sensex_ratio",        # Frontend sends rsc30, uses rsc_sensex_ratio
                "rsc_sensex": "rsc_sensex_ratio",   # Alias
                "rsc_sensex_ratio": "rsc_sensex_ratio",
            }

            series_field = PARAM_FIELD_MAP.get(series_normalized)
            if not series_field:
                print(f"[NRB DEBUG] Unknown series: {series_normalized}")
                return []

            param_qs = (
                Parameter.objects
                .filter(symbol__symbol=scrip)
                .exclude(**{f"{series_field}__isnull": True})
                .order_by("trade_date")
            )

            # DEBUG: Check how many rows we have
            param_count = param_qs.count()
            print(f"[NRB DEBUG] Series: {series_normalized}, Field: {series_field}, Rows: {param_count}")

            weekly_qs = (
                param_qs
                .annotate(week=TruncWeek("trade_date"))
                .values("week")
                .annotate(
                    high=Max(series_field),
                    low=Min(series_field),
                    close=Max(series_field),
                    date=Max("trade_date"),
                )
                .order_by("week")
            )

            total_weeks = weekly_qs.count()
            print(f"[NRB DEBUG] Total weeks: {total_weeks}")

            nr_weeks = weeks if weeks and weeks > 0 else NRB_LOOKBACK
            if total_weeks < nr_weeks + 1:
                print(f"[NRB DEBUG] Not enough weeks: {total_weeks} < {nr_weeks + 1}")
                return []

            weekly_data = list(
                weekly_qs.values("date", "high", "low", "close", "week")
            )
            
            if not weekly_data:
                print("[NRB DEBUG] No weekly data!")
                return []

            # DEBUG: Print first few weeks
            print(f"[NRB DEBUG] First 3 weeks: {weekly_data[:3]}")
            print(f"[NRB DEBUG] Last 3 weeks: {weekly_data[-3:]}")

            # ROLLING-BASED detection with cooldown
            triggers = _detect_narrow_range_break_rolling(weekly_data, nr_weeks, cooldown_weeks)
            print(f"[NRB DEBUG] Triggers found: {len(triggers)}")
            
            triggers = _attach_daily_breakout_times_parameter(
                param_qs, series_field, triggers
            )
            
            # 🆕 Calculate total NRB duration and attach to all triggers
            triggers = _calculate_total_nrb_duration(triggers)

            return triggers

    elif pattern == "Bowl":
        param_qs = (
            Parameter.objects
            .filter(symbol__symbol=scrip)
            .exclude(ema50__isnull=True)
            .order_by("trade_date")
        )
        return _detect_bowl_pattern(param_qs)

    return []


def get_weekly_queryset(base_queryset):
    """
    Converts daily EodPrice rows into weekly OHLC candles.
    """
    return (
        base_queryset.annotate(week=TruncWeek("trade_date"))
        .values("symbol", "week")
        .annotate(
            open=Min("open"),
            high=Max("high"),
            low=Min("low"),
            close=Max("close"),
            date=Max("trade_date"),
        )
        .order_by("week")
    )

def _calculate_total_nrb_duration(triggers):
    """
    Calculate the TOTAL duration from FIRST NRB start to LAST NRB end,
    considering only CONTIGUOUS NRBs that satisfy the 10% buffer condition.
    
    Buffer Logic (Using Breakout Resistance Levels):
    - For each consecutive pair of NRBs, compare their resistance levels (range_high)
    - Calculate 20% of the FIRST NRB's resistance level as the buffer
    - Check if the SECOND NRB's resistance is within 20% of the FIRST
    - This better captures "price level proximity" rather than range overlap
    
    Returns:
    - triggers with added field: total_nrb_duration_weeks
    """
    if not triggers:
        return triggers
    
    # Sort triggers by time to ensure proper ordering
    sorted_triggers = sorted(triggers, key=lambda t: t.get("time", 0))
    
    # Function to check if two NRBs are contiguous (within 10% buffer)
    def are_contiguous(nrb1, nrb2):
        resistance1 = nrb1.get("range_high")  # Resistance broken by NRB #1
        resistance2 = nrb2.get("range_high")  # Resistance broken by NRB #2
        
        if resistance1 is None or resistance2 is None:
            return False
        
        # Calculate 20% buffer based on the first NRB's resistance level
        buffer_threshold = 0.20 * abs(resistance1)
        
        # Calculate the absolute difference between resistance levels
        resistance_diff = abs(resistance2 - resistance1)
        
        # Check if the difference is within 20% of the first resistance
        is_contiguous = resistance_diff <= buffer_threshold
        
        if is_contiguous:
            print(f"[NRB DEBUG] Contiguous: R1={resistance1:.2f}, R2={resistance2:.2f}, diff={resistance_diff:.2f}, buffer={buffer_threshold:.2f}")
        else:
            print(f"[NRB DEBUG] NOT Contiguous: R1={resistance1:.2f}, R2={resistance2:.2f}, diff={resistance_diff:.2f} > buffer={buffer_threshold:.2f}")
        
        return is_contiguous
    
    # Build contiguous groups
    if not sorted_triggers:
        for t in triggers:
            t["total_nrb_duration_weeks"] = None
        return triggers
    
    contiguous_groups = []
    current_group = [sorted_triggers[0]]
    
    for i in range(1, len(sorted_triggers)):
        if are_contiguous(sorted_triggers[i-1], sorted_triggers[i]):
            current_group.append(sorted_triggers[i])
        else:
            contiguous_groups.append(current_group)
            current_group = [sorted_triggers[i]]
    
    # Don't forget the last group
    contiguous_groups.append(current_group)
    
    print(f"[NRB DEBUG] Found {len(contiguous_groups)} contiguous groups")
    for idx, group in enumerate(contiguous_groups):
        if group:
            first_resistance = group[0].get("range_high", 0)
            last_resistance = group[-1].get("range_high", 0)
            print(f"[NRB DEBUG] Group {idx+1}: {len(group)} NRBs, R range: {first_resistance:.2f} to {last_resistance:.2f}")
    
    # Find the group with the longest duration
    longest_group = None
    max_duration_weeks = 0
    
    for group in contiguous_groups:
        if len(group) < 1:
            continue
        
        # Get start and end times for this group
        start_time = group[0].get("range_start_time")
        end_time = group[-1].get("range_end_time")
        
        if start_time is not None and end_time is not None:
            start_date = datetime.fromtimestamp(start_time).date()
            end_date = datetime.fromtimestamp(end_time).date()
            
            total_days = (end_date - start_date).days
            duration_weeks = total_days // 7
            
            if duration_weeks > max_duration_weeks:
                max_duration_weeks = duration_weeks
                longest_group = group
    
    # Calculate duration for the longest contiguous group
    total_duration_weeks = None
    if longest_group and len(longest_group) > 0:
        earliest_start = longest_group[0].get("range_start_time")
        latest_end = longest_group[-1].get("range_end_time")
        
        if earliest_start is not None and latest_end is not None:
            start_date = datetime.fromtimestamp(earliest_start).date()
            end_date = datetime.fromtimestamp(latest_end).date()
            
            total_days = (end_date - start_date).days
            total_duration_weeks = total_days // 7
            
            print(f"[NRB DEBUG] Contiguous NRB Duration: {total_days} days = {total_duration_weeks} weeks")
            print(f"[NRB DEBUG] From {start_date} to {end_date}")
            print(f"[NRB DEBUG] Contiguous group size: {len(longest_group)} NRBs")
    
    # Attach to all triggers
    for t in triggers:
        t["total_nrb_duration_weeks"] = total_duration_weeks
    
    return triggers

def _detect_narrow_range_break_rolling(weekly_data: list, nrb_lookback: int, cooldown_weeks: int = DEFAULT_COOLDOWN_WEEKS):
    """
    ROLLING-BASED NRB Detection with Cooldown Period.
    
    Logic:
    - For each candidate week i (starting from week N onwards)
    - Look back at the PREVIOUS N weeks [i-N, i-1]
    - Find the highest HIGH in those N weeks (resistance)
    - Find the lowest LOW in those N weeks (support)
    - Check if the CURRENT week i breaks above resistance (HIGH > resistance)
    - If yes, record the breakout ONLY if it's beyond the cooldown period from the last breakout
    - Move to next week and repeat (continuous sliding window with cooldown enforcement)
    
    Example with N=52 weeks:
    - Week 52 (index 52): Check if it breaks resistance from weeks 0-51
    - Week 53 (index 53): Check if it breaks resistance from weeks 1-52
    - And so on...
    
    Cooldown Logic:
    - After detecting an NRB at week X, the next NRB can only be detected at week X + cooldown_weeks or later
    - This prevents detecting multiple NRBs in quick succession
    
    For RSC:
    - HIGH = Max(rsc_sensex_ratio) in the week
    - LOW = Min(rsc_sensex_ratio) in the week
    - Resistance = Highest RSC ratio in the N-week window
    - Breakout = Current week's RSC ratio breaks above resistance
    
    Parameters:
    - weekly_data: list of weekly OHLC rows (or RSC high/low for RSC-based NRB)
    - nrb_lookback: N (number of weeks in the rolling window, e.g., 52)
    - cooldown_weeks: Minimum number of weeks between NRB detections (default: 4)
    
    Returns:
    - List of breakout triggers with resistance/support levels (filtered by cooldown)
    """
    
    if len(weekly_data) < nrb_lookback + 1:
        print(f"[NRB DEBUG] Not enough data: {len(weekly_data)} < {nrb_lookback + 1}")
        return []

    rows = [
        {
            "date": row["date"],
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "is_successful_trade": float(row.get("is_successful_trade") or 0.0),
        }
        for row in weekly_data
    ]

    n = len(rows)
    result = []
    nrb_id = 1
    last_breakout_idx = None
    
    # DEBUG counters
    checks_performed = 0
    breakouts_found = 0
    cooldown_blocked = 0

    # Start from week N (index nrb_lookback)
    for i in range(nrb_lookback, n):
        
        checks_performed += 1
        
        # Define the N-week lookback window: [i-N, i-1]
        window_start = i - nrb_lookback
        window_end = i
        
        # Get the N weeks in this window
        window_weeks = rows[window_start:window_end]
        
        # Calculate resistance (highest HIGH) and support (lowest LOW) in this N-week window
        range_high = max(week["high"] for week in window_weeks)
        range_low = min(week["low"] for week in window_weeks)
        
        # Check the CURRENT week (i) for breakout
        breakout_week = rows[i]
        
        # DEBUG: Print first few checks
        if checks_performed <= 3 or (breakout_week["high"] > range_high):
            print(f"[NRB DEBUG] Week {i}: high={breakout_week['high']:.6f}, resistance={range_high:.6f}, breakout={breakout_week['high'] > range_high}")
        
        # NRB condition: Current week's HIGH breaks above N-week resistance
        if breakout_week["high"] > range_high:
            
            breakouts_found += 1
            
            # COOLDOWN CHECK
            if last_breakout_idx is None or (i - last_breakout_idx) >= cooldown_weeks:
                # Breakout found and cooldown satisfied!
                breakout_date = breakout_week["date"]
                score = breakout_week["is_successful_trade"]
                
                trigger_ts = int(
                    datetime.combine(breakout_date, datetime.min.time()).timestamp()
                )
                
                regime_start_ts = int(
                    datetime.combine(rows[window_start]["date"], datetime.min.time()).timestamp()
                )
                regime_end_ts = int(
                    datetime.combine(rows[window_end - 1]["date"], datetime.min.time()).timestamp()
                )
                
                result.append({
                    "time": trigger_ts,
                    "score": score,
                    "direction": "Bullish Break",
                    "range_low": range_low,
                    "range_high": range_high,
                    "range_start_time": regime_start_ts,
                    "range_end_time": regime_end_ts,
                    "nrb_id": nrb_id,
                })
                
                last_breakout_idx = i
                nrb_id += 1
                print(f"[NRB DEBUG] ✓ NRB #{nrb_id-1} detected at week {i}")
            else:
                cooldown_blocked += 1
                print(f"[NRB DEBUG] ✗ Breakout at week {i} blocked by cooldown (last: {last_breakout_idx})")

    print(f"[NRB DEBUG] Summary: checks={checks_performed}, breakouts={breakouts_found}, cooldown_blocked={cooldown_blocked}, result={len(result)}")
    return result


def _attach_daily_breakout_times_price(base_queryset, triggers):
    """
    Refine each NRB trigger from WEEKLY breakout time to the EXACT DAILY candle
    where the DAILY CLOSE first crosses ABOVE the resistance.
    
    This works for price-based (OHLC) NRB detection.
    """
    if not triggers:
        return triggers

    for t in triggers:
        direction = t.get("direction")
        resistance = t.get("range_high")
        weekly_breakout_ts = t.get("time")

        if (
            weekly_breakout_ts is None
            or direction != "Bullish Break"
            or resistance is None
        ):
            continue

        resistance = float(resistance)

        breakout_week_end_date = datetime.fromtimestamp(weekly_breakout_ts).date()
        week_start_date = breakout_week_end_date - timedelta(days=6)

        daily_qs = (
            base_queryset
            .filter(trade_date__gte=week_start_date, trade_date__lte=breakout_week_end_date)
            .order_by("trade_date")
        )

        breakout_daily_date = None

        for row in daily_qs:
            close_f = float(row.close)
            if close_f > resistance:
                breakout_daily_date = row.trade_date
                break

        if breakout_daily_date is not None:
            t["time"] = int(
                datetime.combine(breakout_daily_date, datetime.min.time()).timestamp()
            )

    return triggers


def _attach_daily_breakout_times_parameter(param_qs, value_field: str, triggers):
    """
    Refine each NRB trigger from WEEKLY breakout time to the EXACT DAILY candle
    where the DAILY SERIES value first crosses ABOVE the resistance.
    
    This works for parameter-based (EMA/RSC) NRB detection.
    For RSC: value_field will be 'rsc_sensex_ratio' (the gray line).
    """
    if not triggers:
        return triggers

    for t in triggers:
        direction = t.get("direction")
        resistance = t.get("range_high")
        weekly_breakout_ts = t.get("time")

        if (
            weekly_breakout_ts is None
            or direction != "Bullish Break"
            or resistance is None
        ):
            continue

        resistance = float(resistance)

        breakout_week_end_date = datetime.fromtimestamp(weekly_breakout_ts).date()
        week_start_date = breakout_week_end_date - timedelta(days=6)

        daily_qs = (
            param_qs
            .filter(trade_date__gte=week_start_date, trade_date__lte=breakout_week_end_date)
            .order_by("trade_date")
        )

        breakout_daily_date = None

        for row in daily_qs:
            value = getattr(row, value_field, None)
            if value is None:
                continue
            if float(value) > resistance:
                breakout_daily_date = row.trade_date
                break

        if breakout_daily_date is not None:
            t["time"] = int(
                datetime.combine(breakout_daily_date, datetime.min.time()).timestamp()
            )

    return triggers


def _detect_bowl_pattern(queryset):
    """
    Bowl detection based on EMA 50 stored in Parameter table.
    (Bowl logic remains unchanged)
    """
    raw_rows = list(
        queryset
        .annotate(timestamp=Extract("trade_date", "epoch"))
        .values("trade_date", "timestamp", "ema50", "closing_price")
        .order_by("trade_date")
    )

    rows = [
        {
            "date": r["trade_date"],
            "timestamp": r["timestamp"],
            "ema": float(r["ema50"]),
            "close_f": float(r["closing_price"]),
        }
        for r in raw_rows
    ]

    n = len(rows)
    if n < BOWL_MIN_DURATION_DAYS * 2:
        return []

    result = []
    pattern_id = 1
    last_used = -1

    def clamp(lo, hi):
        return max(lo, 0), min(hi, n - 1)

    for i in range(BOWL_LOCAL_MIN_WINDOW_DAYS, n - BOWL_LOCAL_MIN_WINDOW_DAYS):

        if i <= last_used:
            continue

        ema_i = rows[i]["ema"]

        is_local_min = all(
            rows[j]["ema"] > ema_i
            for j in range(i - BOWL_LOCAL_MIN_WINDOW_DAYS, i + BOWL_LOCAL_MIN_WINDOW_DAYS + 1)
            if j != i
        )

        if not is_local_min:
            continue

        left_start, left_end = clamp(
            i - BOWL_LEFT_LOOKBACK_MAX_DAYS,
            i - BOWL_LEFT_LOOKBACK_MIN_DAYS,
        )
        right_start, right_end = clamp(
            i + BOWL_RIGHT_LOOKAHEAD_MIN_DAYS,
            i + BOWL_RIGHT_LOOKAHEAD_MAX_DAYS,
        )

        if left_end <= left_start or right_end <= right_start:
            continue

        left_slice = rows[left_start:left_end + 1]
        right_slice = rows[right_start:right_end + 1]

        left_idx = left_start + max(
            range(len(left_slice)), key=lambda k: left_slice[k]["ema"]
        )
        right_idx = right_start + max(
            range(len(right_slice)), key=lambda k: right_slice[k]["ema"]
        )

        left_ema = rows[left_idx]["ema"]
        right_ema = rows[right_idx]["ema"]

        total_days = (rows[right_idx]["date"] - rows[left_idx]["date"]).days
        if total_days < BOWL_MIN_TOTAL_DAYS:
            continue

        depth_left = (left_ema - ema_i) / left_ema
        depth_right = (right_ema - ema_i) / right_ema

        if depth_left < BOWL_MIN_DEPTH or depth_right < BOWL_MIN_DEPTH:
            continue

        if min(left_ema, right_ema) / max(left_ema, right_ema) < (1.0 - BOWL_RIM_TOLERANCE):
            continue

        rim_level = max(left_ema, right_ema)
        breakout = None
        for k in range(
            right_idx + 1,
            min(n - 1, right_idx + BOWL_BREAKOUT_LOOKAHEAD_DAYS) + 1,
        ):
            if rows[k]["close_f"] > rim_level:
                breakout = k
                break

        if breakout is None:
            continue

        last_used = breakout
        score = 1.0

        result.append({
            "time": int(rows[left_idx]["timestamp"]),
            "score": score,
            "pattern_id": pattern_id,
        })
        result.append({
            "time": int(rows[i]["timestamp"]),
            "score": score,
            "pattern_id": pattern_id,
        })
        result.append({
            "time": int(rows[right_idx]["timestamp"]),
            "score": score,
            "pattern_id": pattern_id,
        })

        pattern_id += 1

    return result
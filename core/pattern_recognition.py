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

# Buffer percentage for consolidation zone detection (35%)
CONSOLIDATION_BUFFER_PCT = 0.35

# Minimum duration for a valid consolidation zone (in weeks)
MIN_CONSOLIDATION_DURATION_WEEKS = 4


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
      - consolidation_zone_id
      - zone_start_time, zone_end_time
      - zone_duration_weeks
      - zone_min_value, zone_max_value, zone_avg_value

    For Bowl, additional:
      - pattern_id
      
    Parameters:
      - cooldown_weeks: Minimum number of weeks between NRB detections (default: 4)
    
    🆕 UPDATED: Consolidation zones are now detected AFTER NRB detection,
    and only zones that end with an NRB are kept.
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

            # Step 1: Detect NRB breakouts FIRST
            triggers = _detect_narrow_range_break_rolling(weekly_data, nr_weeks, cooldown_weeks)
            triggers = _attach_daily_breakout_times_price(base_queryset, triggers)
            
            # Step 2: Find consolidation zones that END with these NRBs
            consolidation_zones = _find_consolidation_zones_with_nrb(
                weekly_data, 
                triggers,
                series_field='close',
                buffer_pct=CONSOLIDATION_BUFFER_PCT,
                min_duration=MIN_CONSOLIDATION_DURATION_WEEKS
            )
            
            # Step 3: Assign each NRB to its consolidation zone
            triggers = _assign_nrbs_to_zones(triggers, consolidation_zones)

            return triggers

        # CASE 2: PARAMETER-BASED SERIES
        else:
            PARAM_FIELD_MAP = {
                "ema21": "ema21",
                "ema50": "ema50",
                "ema200": "ema200",
                "rsc30": "rsc_sensex_ratio",
                "rsc_sensex": "rsc_sensex_ratio",
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

            print(f"[NRB DEBUG] First 3 weeks: {weekly_data[:3]}")
            print(f"[NRB DEBUG] Last 3 weeks: {weekly_data[-3:]}")

            # Step 1: Detect NRB breakouts FIRST
            triggers = _detect_narrow_range_break_rolling(weekly_data, nr_weeks, cooldown_weeks)
            print(f"[NRB DEBUG] Triggers found: {len(triggers)}")
            
            triggers = _attach_daily_breakout_times_parameter(
                param_qs, series_field, triggers
            )
            
            # Step 2: Find consolidation zones that END with these NRBs
            consolidation_zones = _find_consolidation_zones_with_nrb(
                weekly_data,
                triggers, 
                series_field='close',
                buffer_pct=CONSOLIDATION_BUFFER_PCT,
                min_duration=MIN_CONSOLIDATION_DURATION_WEEKS
            )
            
            # Step 3: Assign each NRB to its consolidation zone
            triggers = _assign_nrbs_to_zones(triggers, consolidation_zones)

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


def _find_consolidation_zones_with_nrb(weekly_data, nrb_triggers, series_field='close', buffer_pct=0.35, min_duration=4):
    """
    🆕 UPDATED: Find consolidation zones that END with an NRB breakout.
    Buffer is calculated based on the FIRST value of each zone.
    
    Algorithm:
    1. Start from each potential zone beginning
    2. Use the FIRST value as the base for buffer calculation
    3. Expand window while values stay within buffer_pct of the first value
    4. A zone is ONLY valid if it ends with an NRB breakout
    5. If no NRB at the end, discard the zone and start fresh from next point
    
    Parameters:
    - weekly_data: List of weekly OHLC/parameter data
    - nrb_triggers: List of NRB breakout triggers (with range_end_time)
    - series_field: 'close' for aggregated weekly value
    - buffer_pct: Maximum percentage range allowed from FIRST value (default: 35%)
    - min_duration: Minimum weeks for a valid zone (default: 4)
    
    Returns:
    - List of valid consolidation zones (only those ending with NRB)
    """
    
    if not weekly_data or not nrb_triggers:
        print("[CONSOLIDATION ZONE DEBUG] No data or no NRB triggers")
        return []
    
    print(f"[CONSOLIDATION ZONE DEBUG] Starting zone detection on {len(weekly_data)} weeks")
    print(f"[CONSOLIDATION ZONE DEBUG] Buffer: {buffer_pct*100}%, Min duration: {min_duration} weeks")
    print(f"[CONSOLIDATION ZONE DEBUG] NRB triggers available: {len(nrb_triggers)}")
    
    # Create a set of NRB end times for quick lookup
    nrb_end_times = set()
    for trigger in nrb_triggers:
        end_time = trigger.get('range_end_time')
        if end_time:
            nrb_end_times.add(end_time)
    
    print(f"[CONSOLIDATION ZONE DEBUG] NRB end times: {len(nrb_end_times)}")
    
    zones = []
    n = len(weekly_data)
    i = 0  # Current starting position
    
    while i < n:
        
        # Get the first value for this potential zone
        first_value = weekly_data[i].get(series_field)
        if first_value is None:
            i += 1
            continue
        
        first_value = float(first_value)
        
        # Calculate buffer bounds based on FIRST value
        lower_bound = first_value * (1 - buffer_pct)
        upper_bound = first_value * (1 + buffer_pct)
        
        print(f"[CONSOLIDATION ZONE DEBUG] Starting zone at week {i}, first_value={first_value:.6f}, bounds=[{lower_bound:.6f}, {upper_bound:.6f}]")
        
        # Expand the zone while values stay within buffer
        j = i
        values_in_zone = [first_value]
        
        while j < n - 1:
            j += 1
            next_value = weekly_data[j].get(series_field)
            
            if next_value is None:
                break
            
            next_value = float(next_value)
            
            # Check if next value is within buffer of FIRST value
            if lower_bound <= next_value <= upper_bound:
                values_in_zone.append(next_value)
            else:
                # Buffer exceeded, stop expanding
                j -= 1  # Step back to last valid point
                break
        
        # Now check if this zone ends with an NRB
        duration = j - i + 1
        
        if duration >= min_duration:
            # Get the end time of this potential zone
            zone_end_date = weekly_data[j].get("date")
            
            if zone_end_date:
                zone_end_ts = int(datetime.combine(zone_end_date, datetime.min.time()).timestamp())
                
                # Check if this zone ends with an NRB
                if zone_end_ts in nrb_end_times:
                    # Valid zone! Save it
                    start_date = weekly_data[i].get("date")
                    start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
                    
                    min_val = min(values_in_zone)
                    max_val = max(values_in_zone)
                    avg_val = sum(values_in_zone) / len(values_in_zone)
                    range_pct = (max_val - min_val) / first_value if first_value > 0 else 0
                    
                    duration_weeks = (zone_end_date - start_date).days / 7.0
                    
                    zone = {
                        'zone_id': len(zones) + 1,
                        'start_idx': i,
                        'end_idx': j,
                        'start_time': start_ts,
                        'end_time': zone_end_ts,
                        'first_value': first_value,
                        'min_value': min_val,
                        'max_value': max_val,
                        'avg_value': avg_val,
                        'duration_weeks': duration_weeks,
                        'range_pct': range_pct * 100  # Convert to percentage
                    }
                    
                    zones.append(zone)
                    
                    print(f"[CONSOLIDATION ZONE DEBUG] ✓ Valid Zone #{zone['zone_id']}: "
                          f"weeks {i}-{j} ({duration_weeks:.1f} weeks), "
                          f"first_value={first_value:.6f}, range {range_pct*100:.1f}%, "
                          f"ends with NRB")
                    
                    # Move to next position AFTER this zone
                    i = j + 1
                else:
                    print(f"[CONSOLIDATION ZONE DEBUG] ✗ Zone at weeks {i}-{j} does NOT end with NRB, discarding")
                    # Zone doesn't end with NRB, start fresh from next point
                    i += 1
            else:
                i += 1
        else:
            # Zone too short, move forward
            print(f"[CONSOLIDATION ZONE DEBUG] ✗ Zone at weeks {i}-{j} too short ({duration} < {min_duration})")
            i += 1
    
    print(f"[CONSOLIDATION ZONE DEBUG] ====== SUMMARY ======")
    print(f"[CONSOLIDATION ZONE DEBUG] Total valid zones found: {len(zones)}")
    for z in zones:
        print(f"[CONSOLIDATION ZONE DEBUG]   Zone {z['zone_id']}: "
              f"{z['duration_weeks']:.1f} weeks, first={z['first_value']:.6f}, "
              f"range {z['range_pct']:.1f}%")
    print(f"[CONSOLIDATION ZONE DEBUG] ====================")
    
    return zones


def _assign_nrbs_to_zones(triggers, consolidation_zones):
    """
    🆕 UPDATED: Assign each NRB to the consolidation zone it belongs to.
    
    Logic:
    1. For each NRB trigger
    2. Check which consolidation zone contains the NRB's consolidation period
    3. Assign the NRB to that zone
    4. Attach zone metadata to the NRB
    
    An NRB belongs to a zone if its consolidation period (range_start_time to range_end_time)
    overlaps with the zone's time period.
    
    Parameters:
    - triggers: List of NRB trigger dicts
    - consolidation_zones: List of consolidation zone dicts
    
    Returns:
    - triggers with zone metadata attached
    """
    
    if not triggers or not consolidation_zones:
        print("[NRB ZONE ASSIGNMENT] No triggers or zones to assign")
        return triggers
    
    print(f"[NRB ZONE ASSIGNMENT] Assigning {len(triggers)} NRBs to {len(consolidation_zones)} zones")
    
    for trigger in triggers:
        # Get the NRB's consolidation period
        nrb_range_start = trigger.get("range_start_time")
        nrb_range_end = trigger.get("range_end_time")
        nrb_breakout_time = trigger.get("time")
        
        if not nrb_range_start or not nrb_range_end:
            continue
        
        # Find the zone that overlaps with this NRB's consolidation period
        best_zone = None
        best_overlap = 0
        
        for zone in consolidation_zones:
            zone_start = zone['start_time']
            zone_end = zone['end_time']
            
            # Check for overlap
            overlap_start = max(nrb_range_start, zone_start)
            overlap_end = min(nrb_range_end, zone_end)
            
            if overlap_start <= overlap_end:
                # There is overlap
                overlap_duration = overlap_end - overlap_start
                
                if overlap_duration > best_overlap:
                    best_overlap = overlap_duration
                    best_zone = zone
        
        # Assign to the best matching zone
        if best_zone:
            trigger['consolidation_zone_id'] = best_zone['zone_id']
            trigger['zone_start_time'] = best_zone['start_time']
            trigger['zone_end_time'] = best_zone['end_time']
            trigger['zone_duration_weeks'] = best_zone['duration_weeks']
            trigger['zone_min_value'] = best_zone['min_value']
            trigger['zone_max_value'] = best_zone['max_value']
            trigger['zone_avg_value'] = best_zone['avg_value']
            trigger['zone_range_pct'] = best_zone['range_pct']
            trigger['zone_first_value'] = best_zone['first_value']
            
            nrb_id = trigger.get('nrb_id', '?')
            print(f"[NRB ZONE ASSIGNMENT] NRB #{nrb_id} → Zone #{best_zone['zone_id']} "
                  f"(overlap: {best_overlap / (7*24*60*60):.1f} weeks)")
        else:
            print(f"[NRB ZONE ASSIGNMENT] NRB #{trigger.get('nrb_id', '?')} has no matching zone")
    
    # Group statistics
    zone_nrb_counts = {}
    for trigger in triggers:
        zone_id = trigger.get('consolidation_zone_id')
        if zone_id:
            zone_nrb_counts[zone_id] = zone_nrb_counts.get(zone_id, 0) + 1
    
    print(f"[NRB ZONE ASSIGNMENT] ====== ASSIGNMENT SUMMARY ======")
    for zone_id, count in sorted(zone_nrb_counts.items()):
        print(f"[NRB ZONE ASSIGNMENT]   Zone {zone_id}: {count} NRBs")
    print(f"[NRB ZONE ASSIGNMENT] ==================================")
    
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
    
    Parameters:
    - weekly_data: list of weekly OHLC rows
    - nrb_lookback: N (number of weeks in the rolling window)
    - cooldown_weeks: Minimum number of weeks between NRB detections
    
    Returns:
    - List of breakout triggers (filtered by cooldown)
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
    
    checks_performed = 0
    breakouts_found = 0
    cooldown_blocked = 0

    for i in range(nrb_lookback, n):
        
        checks_performed += 1
        
        window_start = i - nrb_lookback
        window_end = i
        
        window_weeks = rows[window_start:window_end]
        
        range_high = max(week["high"] for week in window_weeks)
        range_low = min(week["low"] for week in window_weeks)
        
        breakout_week = rows[i]
        
        if checks_performed <= 3 or (breakout_week["high"] > range_high):
            print(f"[NRB DEBUG] Week {i}: high={breakout_week['high']:.6f}, resistance={range_high:.6f}, breakout={breakout_week['high'] > range_high}")
        
        if breakout_week["high"] > range_high:
            
            breakouts_found += 1
            
            if last_breakout_idx is None or (i - last_breakout_idx) >= cooldown_weeks:
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
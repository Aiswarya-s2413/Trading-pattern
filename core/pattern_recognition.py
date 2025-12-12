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



def get_pattern_triggers(
    scrip: str,
    pattern: str,
    nrb_lookback: int,
    success_rate: float,
    weeks: int = 20,
    series: str | None = None,
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

    For Bowl, additional:
      - pattern_id
    """
    series_normalized = (series or "").strip().lower()

    if pattern == "Narrow Range Break":

        # CASE 1: DEFAULT – use price candles (EodPrice)
        if series_normalized in ("", "price", "close", "closing_price"):
            base_queryset = EodPrice.objects.filter(symbol__symbol=scrip)
            weekly_qs = get_weekly_queryset(base_queryset)
            total_weeks = weekly_qs.count()

            nr_weeks = weeks if weeks and weeks > 0 else NRB_LOOKBACK

            if total_weeks < nr_weeks + 2:
                return []

            weekly_data = list(
                weekly_qs.values(
                    "date", "high", "low", "close", "week"
                )
            )

            if not weekly_data:
                return []

            # ROLLING-BASED detection
            triggers = _detect_narrow_range_break_rolling(weekly_data, nr_weeks)
            triggers = _attach_daily_breakout_times_price(base_queryset, triggers)

            return triggers

        # CASE 2: PARAMETER-BASED SERIES
        else:
            PARAM_FIELD_MAP = {
                "ema21": "ema21",
                "ema50": "ema50",
                "ema200": "ema200",
                "rsc30": "rsc30",
                "rsc500": "rsc500",
            }

            series_field = PARAM_FIELD_MAP.get(series_normalized)
            if not series_field:
                return []

            param_qs = (
                Parameter.objects
                .filter(symbol__symbol=scrip)
                .exclude(**{f"{series_field}__isnull": True})
                .order_by("trade_date")
            )

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

            nr_weeks = weeks if weeks and weeks > 0 else NRB_LOOKBACK
            if total_weeks < nr_weeks + 2:
                return []

            weekly_data = list(
                weekly_qs.values("date", "high", "low", "close", "week")
            )
            if not weekly_data:
                return []

            # ROLLING-BASED detection
            triggers = _detect_narrow_range_break_rolling(weekly_data, nr_weeks)
            triggers = _attach_daily_breakout_times_parameter(
                param_qs, series_field, triggers
            )

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


def _detect_narrow_range_break_rolling(weekly_data: list, nrb_lookback: int):
    """
    ROLLING-BASED NRB Detection.
    
    Logic:
    - For each week i (starting from week N), look back at the last N weeks [i-N+1 to i]
    - Find the highest HIGH in those N weeks (resistance)
    - Find the lowest LOW in those N weeks (support)
    - Check if the VERY NEXT week (i+1) breaks above resistance (HIGH > resistance)
    - If yes, record the breakout
    - Move to next week and repeat (no cooldown, continuous sliding window)
    
    Example with N=52:
    - Week 52: Check weeks 1-52, resistance = highest HIGH in weeks 1-52
              If week 53 breaks above, record it
    - Week 53: Check weeks 2-53, resistance = highest HIGH in weeks 2-53
              If week 54 breaks above, record it
    - Week 54: Check weeks 3-54, and so on...
    
    Parameters:
    - weekly_data: list of weekly OHLC rows
    - nrb_lookback: N (number of weeks in the rolling window, e.g., 52)
    
    Returns:
    - List of breakout triggers with resistance/support levels
    """
    
    if len(weekly_data) < nrb_lookback + 1:
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

    # Start from week N (index N-1 in 0-based indexing)
    # We need at least 1 more week after the N-week window for breakout check
    for i in range(nrb_lookback - 1, n - 1):
        
        # Define the N-week window: [i-N+1, i]
        window_start = i - nrb_lookback + 1
        window_end = i + 1  # exclusive in Python slicing
        
        if window_start < 0:
            continue
        
        # Get the N weeks in this window
        window_weeks = rows[window_start:window_end]
        
        # Calculate resistance (highest HIGH) and support (lowest LOW) in this N-week window
        range_high = max(week["high"] for week in window_weeks)
        range_low = min(week["low"] for week in window_weeks)
        
        # Check the NEXT week (i+1) for breakout
        breakout_idx = i + 1
        
        if breakout_idx >= n:
            # No next week available
            continue
        
        breakout_week = rows[breakout_idx]
        
        # NRB condition: Next week's HIGH breaks above N-week resistance
        if breakout_week["high"] > range_high:
            # Breakout found!
            breakout_date = breakout_week["date"]
            score = breakout_week["is_successful_trade"]
            
            trigger_ts = int(
                datetime.combine(breakout_date, datetime.min.time()).timestamp()
            )
            
            regime_start_ts = int(
                datetime.combine(rows[window_start]["date"], datetime.min.time()).timestamp()
            )
            regime_end_ts = int(
                datetime.combine(breakout_date, datetime.min.time()).timestamp()
            )
            
            result.append({
                "time": trigger_ts,  # Breakout week timestamp
                "score": score,
                "direction": "Bullish Break",
                "range_low": range_low,
                "range_high": range_high,
                "range_start_time": regime_start_ts,
                "range_end_time": regime_end_ts,
                "nrb_id": nrb_id,
            })
            
            nrb_id += 1

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
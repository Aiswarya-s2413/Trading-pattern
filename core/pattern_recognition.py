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
    nrb_lookback: int,   # kept for backward-compatibility; ignored for NRB now
    success_rate: float,
    weeks: int = 20,     # <-- this is the "N" in NRn (NR4, NR7, NR52,...)
    series: str | None = None,  # <-- which series to use for NRB (None = price)
):
    """
    Main entry to compute pattern triggers for a given symbol and pattern type.
    Returns a list of dicts; each dict at minimum has:
      - time (unix ts in seconds)
      - score (float)

    For NRB, additional fields are attached:
      - direction: "Bullish Break"
      - range_low, range_high                # N-week regime support/resistance
      - range_start_time, range_end_time     # regime time span
      - nrb_id                               # regime identifier
      - nr_high, nr_low                      # NR week’s own high/low

    For Bowl, additional:
      - pattern_id

    SERIES BEHAVIOR (NRB only):

      - series is None / "price" / "close":
          * Use daily OHLC from EodPrice (default).
          * NRB is based on price candles and breakout above N-week HIGH.

      - series in {"ema21", "ema50", "ema200", "rsc30", "rsc500"}:
          * Use daily values from Parameter table.
          * Weekly regime is built from that series.
          * Breakout is when that series crosses above its N-week high.
    """
    # Normalize series string
    series_normalized = (series or "").strip().lower()

   
    #   NARROW RANGE BREAK (NRn)
   
    if pattern == "Narrow Range Break":

        # CASE 1: DEFAULT – use price candles (EodPrice)
        if series_normalized in ("", "price", "close", "closing_price"):
            # NOTE: symbol is a ForeignKey; scrip is symbol string (e.g. "RELIANCE.NS")
            base_queryset = EodPrice.objects.filter(symbol__symbol=scrip)

            # Build weekly candles (price-based)
            weekly_qs = get_weekly_queryset(base_queryset)
            total_weeks = weekly_qs.count()

            # n = number of WEEKLY candles we look at for NRn.
            # This comes from frontend `weeks`, with a fallback.
            nr_weeks = weeks if weeks and weeks > 0 else NRB_LOOKBACK

            # Need at least n weekly candles + 1 breakout candle
            if total_weeks < nr_weeks + 2:
                return []

            weekly_data = list(
                weekly_qs.values(
                    "date", "high", "low", "close", "week"
                )
            )

            if not weekly_data:
                return []

            # Step 1: detect NRn + breakout on WEEKLY data (price-based)
            triggers = _detect_narrow_range_break_python(weekly_data, nr_weeks)

            # Step 2: refine each NRB to the EXACT DAILY breakout candle
            #         where daily CLOSE first > N-week resistance (range_high).
            triggers = _attach_daily_breakout_times_price(base_queryset, triggers)

            # Filter by success rate (we now default score=0.0, so success_rate>0 will filter all out)
            return triggers

        # CASE 2: PARAMETER-BASED SERIES (EMA21 / EMA50 / EMA200 / RSC30 / RSC500)
        else:
            # Map acceptable series names to Parameter model fields
            PARAM_FIELD_MAP = {
                "ema21": "ema21",
                "ema50": "ema50",
                "ema200": "ema200",
                "rsc30": "rsc30",
                "rsc500": "rsc500",
            }

            series_field = PARAM_FIELD_MAP.get(series_normalized)
            if not series_field:
                # Unknown series → gracefully fallback to no triggers
                return []

            # Parameter-based queryset for this symbol
            param_qs = (
                Parameter.objects
                .filter(symbol__symbol=scrip)
                .exclude(**{f"{series_field}__isnull": True})
                .order_by("trade_date")
            )

            # Build weekly "candles" for the chosen series (from Parameter)
            weekly_qs = (
                param_qs
                .annotate(week=TruncWeek("trade_date"))
                .values("week")
                .annotate(
                    high=Max(series_field),
                    low=Min(series_field),
                    close=Max(series_field),  # not really used in logic
                    date=Max("trade_date"),   # last date in that week
                    # no is_successful_trade here → score will default to 0.0
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

            # Step 1: detect NRn + breakout on WEEKLY data (series-based)
            triggers = _detect_narrow_range_break_python(weekly_data, nr_weeks)

            # Step 2: refine each NRB to the EXACT DAILY breakout candle
            #         where the DAILY SERIES value first > N-week resistance.
            triggers = _attach_daily_breakout_times_parameter(
                param_qs, series_field, triggers
            )

            # For Parameter-series, we currently ignore success_rate (no score field).
            return triggers

  
    # BOWL

    elif pattern == "Bowl":
        # For Bowl, we now base EMA50 on Parameter table
        param_qs = (
            Parameter.objects
            .filter(symbol__symbol=scrip)
            .exclude(ema50__isnull=True)
            .order_by("trade_date")
        )
        return _detect_bowl_pattern(param_qs)

    return []


#  WEEKLY CANDLE BUILDER


def get_weekly_queryset(base_queryset):
    """
    Converts daily EodPrice rows into weekly OHLC candles.
    Each row in the result represents ONE WEEK.
    """
    return (
        base_queryset.annotate(week=TruncWeek("trade_date"))
        .values("symbol", "week")
        .annotate(
            open=Min("open"),
            high=Max("high"),
            low=Min("low"),
            close=Max("close"),
            date=Max("trade_date"),  # last date in that week
            # is_successful_trade removed in new schema; score defaults to 0.0
        )
        .order_by("week")
    )


#  NARROW RANGE LOGIC


def _detect_narrow_range_break_python(weekly_data: list, nrb_lookback: int):
    """
    Detects NRn (Narrow Range n) on WEEKLY data, in a *regime* style.
    Intended logic:

      - Regime = last n weekly candles.
      - Resistance = highest HIGH in that n-week regime.
        For n=52, this is the 52-week high.
      - NRn week = the week in that regime with the SMALLEST range (high-low).
      - Breakout = the FIRST WEEK (within a small lookahead window) whose HIGH
                   crosses ABOVE that resistance (high > resistance).
                   Only upside, no bearish breaks.

      - After a breakout, that regime is OVER.
        We must then wait another full n weeks before looking for a new NRn.

    NOTE: This function returns breakout time as the WEEKLY breakout candle’s
    date. A separate helper refines this to the EXACT DAILY candle where the
    DAILY CLOSE/value first > resistance.

    - weekly_data: list of weekly OHLC-like rows (each row = one week)
    - nrb_lookback: n in NRn (e.g. 4, 7, 10, 52)
    """

    # Need at least n weekly candles + 1 breakout candle
    if len(weekly_data) < nrb_lookback + 1:
        return []

    preceding_rows = max(nrb_lookback - 1, 0)
    result = []

    rows = [
        {
            "date": row["date"],
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            # new schema has no is_successful_trade -> default to 0.0
            "is_successful_trade": float(row.get("is_successful_trade") or 0.0),
        }
        for row in weekly_data
    ]

    n = len(rows)

    # Index of the last breakout candle (by WEEK index).
    # We enforce a cooldown of n weeks after this index.
    last_breakout_idx = -1
    nrb_id = 1  # identifier to group each narrow-range regime

    # How many weeks we allow for breakout after the NR week
    BREAKOUT_LOOKAHEAD_WEEKS = 12

    for i in range(preceding_rows, n - 1):
        # REGIME / COOLDOWN LOGIC
        # Ensure the full n-week window [i-(n-1) .. i] is AFTER the last breakout.
        if last_breakout_idx != -1 and i < last_breakout_idx + nrb_lookback:
            # Still inside cooldown → skip
            continue

        # Range of the current NRn candidate week (NR week)
        nr_high = rows[i]["high"]
        nr_low = rows[i]["low"]
        spread = nr_high - nr_low

        # Window of the last n weeks (including i)
        window_start = max(0, i - preceding_rows)
        window_end = i + 1  # inclusive of i

        # Compute narrowest range in that n-week window
        # And the regime's support/resistance (min low, max high)
        min_spread = float("inf")
        range_high = float("-inf")  # resistance (N-week high)
        range_low = float("inf")    # support  (N-week low)

        for j in range(window_start, window_end):
            r = rows[j]
            r_spread = r["high"] - r["low"]
            if r_spread < min_spread:
                min_spread = r_spread
            if r["high"] > range_high:
                range_high = r["high"]
            if r["low"] < range_low:
                range_low = r["low"]

        # We only care if current week has the smallest range in the last n weeks
        if spread != min_spread:
            continue

        # We have an NRn week at index i.
        # Now look ahead up to BREAKOUT_LOOKAHEAD_WEEKS to find first HIGH > resistance.
        resistance = range_high  # N-week high
        breakout_idx = None

        # Look ahead from i+1 to i+BREAKOUT_LOOKAHEAD_WEEKS
        lookahead_end = min(n, i + 1 + BREAKOUT_LOOKAHEAD_WEEKS)
        for k in range(i + 1, lookahead_end):
            if rows[k]["high"] > resistance:
                breakout_idx = k
                break

        if breakout_idx is None:
            # No breakout in the lookahead window → skip this NRn
            continue

        breakout_row = rows[breakout_idx]
        breakout_date = breakout_row["date"]
        candle_break = "Bullish Break"  # only bullish breakouts
        score = breakout_row["is_successful_trade"]

        trigger_ts = int(
            datetime.combine(breakout_date, datetime.min.time()).timestamp()
        )

        # Regime covers the n-week window (for drawing horizontal lines).
        regime_start_ts = int(
            datetime.combine(rows[window_start]["date"], datetime.min.time()).timestamp()
        )
        # We end at the breakout week for a nice visual box up to breakout.
        regime_end_ts = int(
            datetime.combine(breakout_date, datetime.min.time()).timestamp()
        )

        result.append({
            # NOTE: this is the WEEKLY breakout time,
            # will be refined to DAILY in _attach_daily_breakout_times_* helpers.
            "time": trigger_ts,
            "score": score,
            "direction": candle_break,
            # extra fields for plotting the narrow range box/lines
            "range_low": range_low,
            "range_high": range_high,
            "range_start_time": regime_start_ts,
            "range_end_time": regime_end_ts,
            "nrb_id": nrb_id,
            # store NR week high/low (for debugging / future use)
            "nr_high": nr_high,
            "nr_low": nr_low,
        })

        # Mark this breakout index and start a new regime AFTER n more weeks.
        last_breakout_idx = breakout_idx
        nrb_id += 1

    return result



#   DAILY REFINEMENT HELPERS

def _attach_daily_breakout_times_price(base_queryset, triggers):
    """
    Refine each NRB trigger (price-based) from WEEKLY breakout time to the
    EXACT DAILY candle inside that breakout week where the DAILY CLOSE first
    crosses ABOVE the N-week resistance (range_high).

    - base_queryset: EodPrice.objects.filter(symbol__symbol=...)
    - triggers: list of dicts returned from _detect_narrow_range_break_python
    """
    if not triggers:
        return triggers

    for t in triggers:
        direction = t.get("direction")
        # Use the regime resistance level (N-week high) as breakout threshold
        resistance = t.get("range_high")
        weekly_breakout_ts = t.get("time")

        if (
            weekly_breakout_ts is None
            or direction != "Bullish Break"
            or resistance is None
        ):
            continue

        resistance = float(resistance)

        # Convert weekly breakout ts -> date (end of that breakout week)
        breakout_week_end_date = datetime.fromtimestamp(weekly_breakout_ts).date()

        # Approximate start of that week as 6 days before the end date
        # (covers Mon-Sun; trading days will be subset)
        week_start_date = breakout_week_end_date - timedelta(days=6)

        # Fetch all DAILY candles in that breakout week
        daily_qs = (
            base_queryset
            .filter(trade_date__gte=week_start_date, trade_date__lte=breakout_week_end_date)
            .order_by("trade_date")
        )

        breakout_daily_date = None

        # Bullish NRB: first daily CLOSE > resistance (N-week high)
        for row in daily_qs:
            close_f = float(row.close)
            if close_f > resistance:
                breakout_daily_date = row.trade_date
                break

        # If we found a daily breakout, update the trigger time
        if breakout_daily_date is not None:
            t["time"] = int(
                datetime.combine(breakout_daily_date, datetime.min.time()).timestamp()
            )

    return triggers


def _attach_daily_breakout_times_parameter(param_qs, value_field: str, triggers):
    """
    Refine each NRB trigger (Parameter-based series) from WEEKLY breakout time
    to the EXACT DAILY candle inside that breakout week where the DAILY
    SERIES value first crosses ABOVE the N-week resistance (range_high).

    - param_qs: Parameter.objects.filter(symbol__symbol=..., [series_field not null])
    - value_field: one of {"ema21", "ema50", "ema200", "rsc30", "rsc500"}
    - triggers: list of dicts returned from _detect_narrow_range_break_python
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

        # Fetch all DAILY Parameter rows in that breakout week
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



#  BOWL PATTERN


def _detect_bowl_pattern(queryset):
    """
    Bowl detection based on EMA 50 stored in Parameter table.

    - queryset: Parameter.objects.filter(symbol__symbol=..., ema50__isnull=False)
    """
    # Build a normalized list of rows with consistent keys
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

        # Local bottom check
        is_local_min = all(
            rows[j]["ema"] > ema_i
            for j in range(i - BOWL_LOCAL_MIN_WINDOW_DAYS, i + BOWL_LOCAL_MIN_WINDOW_DAYS + 1)
            if j != i
        )

        if not is_local_min:
            continue

        # Find rims (left & right high points in EMA)
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

        # Total duration of the bowl
        total_days = (rows[right_idx]["date"] - rows[left_idx]["date"]).days
        if total_days < BOWL_MIN_TOTAL_DAYS:
            continue

        # Depth check
        depth_left = (left_ema - ema_i) / left_ema
        depth_right = (right_ema - ema_i) / right_ema

        if depth_left < BOWL_MIN_DEPTH or depth_right < BOWL_MIN_DEPTH:
            continue

        # Rim symmetry
        if min(left_ema, right_ema) / max(left_ema, right_ema) < (1.0 - BOWL_RIM_TOLERANCE):
            continue

        # Breakout above rim
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
        # new schema has no is_successful_trade -> fixed score
        score = 1.0

        # Mark three points of the bowl: left rim, bottom, right rim
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

from datetime import datetime

from django.db.models import F, Q, Window, Min
from django.db.models.expressions import RowRange
from django.db.models.functions import Extract, Lead
from api.models import PriceFeature

# --- Constants for Logic Checks ---
BOWL_MIN_DURATION_DAYS = 60  # Minimum days required to form a Bowl pattern

# --- Bowl Detection (rounded bottom) ---
# Local minimum check: bottom must be lowest vs ±window days
BOWL_LOCAL_MIN_WINDOW_DAYS = 2  # relaxed to detect more bottoms

# How far we look to the left/right of the bottom to find rims
BOWL_LEFT_LOOKBACK_MIN_DAYS = 20
BOWL_LEFT_LOOKBACK_MAX_DAYS = 90
BOWL_RIGHT_LOOKAHEAD_MIN_DAYS = 20
BOWL_RIGHT_LOOKAHEAD_MAX_DAYS = 90

# Minimum % drop from rim to bottom (e.g. 0.06 = 6%)
BOWL_MIN_DEPTH = 0.06  # relaxed from 10% to 6%

# Left & right rims must be within this tolerance of each other
# 0.3 = 30% difference allowed
BOWL_RIM_TOLERANCE = 0.30

# Total bowl duration (left rim → right rim)
BOWL_MIN_TOTAL_DAYS = 40

# How far after the right rim we search for breakout
BOWL_BREAKOUT_LOOKAHEAD_DAYS = 120  # extended from 60


def get_pattern_triggers(
    scrip: str, pattern: str, nrb_lookback: int, success_rate: float
):
    """
    The main routing function to execute pattern-specific database queries.
    """
    base_queryset = PriceFeature.objects.filter(symbol=scrip)

    if pattern == "Narrow Range Break":
        # Detect NRB first, then filter by BREAKOUT score (not setup candle)
        triggers = _detect_narrow_range_break(base_queryset, nrb_lookback)
        return [t for t in triggers if t.get("score", 0.0) >= success_rate]

    elif pattern == "Bowl":
        # For Bowl, we need all historical data to detect the pattern
        # Success rate filtering is disabled or can be handled separately
        triggers = _detect_bowl_pattern(base_queryset)
        return triggers

    return []





def _detect_narrow_range_break(queryset, nrb_lookback: int):
    """
    Detects Narrow Range Break using dynamic lookback (N) via ORM window functions.
    A day qualifies as NRB when its range is the tightest in the trailing window **and**
    the next day's close breaks above the high (bullish) or below the low (bearish).
    """
    preceding_rows = max(nrb_lookback - 1, 0)
    window_frame = RowRange(start=-preceding_rows, end=0)

    triggers = (
        queryset.annotate(spread=F("high") - F("low"))
        .annotate(
            min_spread=Window(
                expression=Min("spread"),
                partition_by=["symbol"],
                order_by=["date"],
                frame=window_frame,
            )
        )
        .filter(spread=F("min_spread"))
        .annotate(
            next_close=Window(
                expression=Lead("close"), partition_by=["symbol"], order_by=["date"]
            ),
            next_high=Window(
                expression=Lead("high"), partition_by=["symbol"], order_by=["date"]
            ),
            next_low=Window(
                expression=Lead("low"), partition_by=["symbol"], order_by=["date"]
            ),
            next_date=Window(
                expression=Lead("date"), partition_by=["symbol"], order_by=["date"]
            ),
            trigger_score=Window(
                expression=Lead("is_successful_trade"),
                partition_by=["symbol"],
                order_by=["date"],
            ),
        )
        .filter(next_close__isnull=False)
        .filter(Q(next_close__gt=F("high")) | Q(next_close__lt=F("low")))
        .values("date", "next_date", "high", "low", "next_close", "trigger_score")
    )

    # Final Formatting in Python
    return [_format_nrb_trigger(row) for row in triggers]


def _format_nrb_trigger(row):
    """
    Convert ORM row to plotting payload. Keeps logic isolated for clarity.
    """
    trigger_date = row["next_date"]
    candle_break = (
        "Bullish Break" if row["next_close"] > row["high"] else "Bearish Break"
    )

    raw_score = row.get("trigger_score")
    score = float(raw_score) if raw_score is not None else 0.0

    return {
        "time": int(datetime.combine(trigger_date, datetime.min.time()).timestamp()),
        "score": score,
        "direction": candle_break,
    }




def _detect_bowl_pattern(queryset):
    """
    Proper rounded-bottom (bowl / cup) detector.

    Algorithm (EMA_50-based):

    1. Pull ordered price series (date, EMA_50, high, close, is_successful_trade).

    2. Find candidate bottoms where EMA_50 is a local minimum vs ±BOWL_LOCAL_MIN_WINDOW_DAYS.

    3. For each bottom:
       - Search LEFT for a rim: max EMA_50 in [bottom - 90d, bottom - 20d].
       - Search RIGHT for a rim: max EMA_50 in [bottom + 20d, bottom + 90d].

    4. Apply bowl constraints:
       - Depth from both rims ≥ BOWL_MIN_DEPTH (e.g. ≥ 6%).
       - Rims are similar: within BOWL_RIM_TOLERANCE (e.g. ≤ 30% difference).
       - Total duration left_rim → right_rim ≥ BOWL_MIN_TOTAL_DAYS.

    5. Confirm breakout:
       - Look ahead up to BOWL_BREAKOUT_LOOKAHEAD_DAYS after right rim.
       - First close > max(left_rim_ema, right_rim_ema) is considered breakout.

    6. For each valid bowl, return 3 markers:
       - left rim, bottom, right rim; all sharing the same pattern_id.
    """

    # Step 1: load series into Python (ordered by date)
    rows = list(
        queryset.filter(EMA_50__isnull=False)
        .annotate(timestamp=Extract("date", "epoch"))
        .values("date", "timestamp", "EMA_50", "high", "close", "is_successful_trade")
        .order_by("date")
    )

    n = len(rows)
    if n < BOWL_MIN_DURATION_DAYS * 2:
        return []

    # convenience: convert EMA & close to float for calculations
    for row in rows:
        row["ema"] = float(row["EMA_50"])
        row["close_f"] = float(row["close"])

    result = []
    pattern_id = 1
    last_used_index = -1  # to avoid overlapping bowls

    # Helper to clamp index range
    def clamp(lo, hi):
        return max(lo, 0), min(hi, n - 1)

    # Step 2: scan for local EMA bottoms
    for i in range(BOWL_LOCAL_MIN_WINDOW_DAYS, n - BOWL_LOCAL_MIN_WINDOW_DAYS):
        if i <= last_used_index:
            continue  # skip inside an already-used bowl

        ema_i = rows[i]["ema"]

        # Local minimum check within ±window
        is_local_min = True
        for j in range(
            i - BOWL_LOCAL_MIN_WINDOW_DAYS,
            i + BOWL_LOCAL_MIN_WINDOW_DAYS + 1,
        ):
            if j == i:
                continue
            if rows[j]["ema"] <= ema_i:
                is_local_min = False
                break

        if not is_local_min:
            continue

        # Step 3: find LEFT rim (max EMA in [i-90, i-20])
        left_start, left_end = clamp(
            i - BOWL_LEFT_LOOKBACK_MAX_DAYS,
            i - BOWL_LEFT_LOOKBACK_MIN_DAYS,
        )
        if left_end <= left_start:
            continue  # not enough history

        left_slice = rows[left_start : left_end + 1]
        left_idx_rel = max(range(len(left_slice)), key=lambda k: left_slice[k]["ema"])
        left_idx = left_start + left_idx_rel
        left_ema = left_slice[left_idx_rel]["ema"]

        # Step 4: find RIGHT rim (max EMA in [i+20, i+90])
        right_start, right_end = clamp(
            i + BOWL_RIGHT_LOOKAHEAD_MIN_DAYS,
            i + BOWL_RIGHT_LOOKAHEAD_MAX_DAYS,
        )
        if right_end <= right_start:
            continue  # not enough future

        right_slice = rows[right_start : right_end + 1]
        right_idx_rel = max(
            range(len(right_slice)),
            key=lambda k: right_slice[k]["ema"],
        )
        right_idx = right_start + right_idx_rel
        right_ema = right_slice[right_idx_rel]["ema"]

        # Step 5: bowl constraints

        # Durations
        total_days = (rows[right_idx]["date"] - rows[left_idx]["date"]).days
        if total_days < BOWL_MIN_TOTAL_DAYS:
            continue

        # Depth from rims
        depth_left = (left_ema - ema_i) / left_ema if left_ema > 0 else 0.0
        depth_right = (right_ema - ema_i) / right_ema if right_ema > 0 else 0.0

        if depth_left < BOWL_MIN_DEPTH or depth_right < BOWL_MIN_DEPTH:
            continue

        # Rims similar height (symmetry-ish)
        rim_max = max(left_ema, right_ema)
        rim_min = min(left_ema, right_ema)
        if rim_min / rim_max < (1.0 - BOWL_RIM_TOLERANCE):
            continue

        # Optional: ensure EMA is generally rising from bottom to right rim
        up_moves = 0
        total_moves = 0
        for k in range(i + 1, right_idx + 1):
            if rows[k]["ema"] > rows[k - 1]["ema"]:
                up_moves += 1
            total_moves += 1
        if total_moves == 0 or up_moves / total_moves < 0.5:
            # less than 50% up moves → too choppy
            continue

        # Step 6: breakout – close > rim level after right rim
        rim_level = rim_max
        breakout_idx = None
        breakout_limit_idx = min(n - 1, right_idx + BOWL_BREAKOUT_LOOKAHEAD_DAYS)
        for k in range(right_idx + 1, breakout_limit_idx + 1):
            if rows[k]["close_f"] > rim_level:
                breakout_idx = k
                break

        if breakout_idx is None:
            # unconfirmed bowl (no breakout)
            continue

        # Accept this bowl; avoid overlapping with later ones
        last_used_index = breakout_idx

        # Pattern "score" – use success at breakout; default to 1.0 if unlabeled
        raw_score = rows[breakout_idx].get("is_successful_trade")
        score = float(raw_score) if raw_score is not None else 1.0

        # Step 7: push markers for frontend (left rim, bottom, right rim)
        left_ts = int(rows[left_idx]["timestamp"])
        bottom_ts = int(rows[i]["timestamp"])
        right_ts = int(rows[right_idx]["timestamp"])

        result.append({"time": left_ts, "score": score, "pattern_id": pattern_id})
        result.append({"time": bottom_ts, "score": score, "pattern_id": pattern_id})
        result.append({"time": right_ts, "score": score, "pattern_id": pattern_id})

        pattern_id += 1

    return result

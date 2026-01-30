from datetime import datetime, date, timedelta

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

# Tolerance for grouping NRBs at same level (5% by default)
NRB_LEVEL_TOLERANCE_PCT = 0.05

# Tolerance for identifying "Near Touch" zones (10%)
NEAR_TOUCH_TOLERANCE_PCT = 0.10

# Maximum allowed gap (in days) between touches to maintain the "Historical Line"
MAX_HISTORY_GAP_DAYS = 8000


def get_pattern_triggers(
    scrip: str,
    pattern: str,
    nrb_lookback: int,
    success_rate: float,
    weeks: int = 20,
    series: str | None = None,
    cooldown_weeks: int = DEFAULT_COOLDOWN_WEEKS,
    dip_threshold_pct: float = 0.20,
    whipsaw_d1: int | None = None,
    whipsaw_d2: int | None = None,
):
    """
    Main entry to compute pattern triggers for a given symbol and pattern type.
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

            # 1. Get All-Time High/Low for Deep Dip calculation
            stats = base_queryset.aggregate(min_val=Min('low'), max_val=Max('high'))
            global_min = float(stats['min_val']) if stats['min_val'] else 0
            global_max = float(stats['max_val']) if stats['max_val'] else 0

            # Step 1: Detect NRB breakouts FIRST
            triggers = _detect_narrow_range_break_rolling(weekly_data, nr_weeks, cooldown_weeks)
            triggers = _attach_daily_breakout_times_price(base_queryset, triggers)
            
            # Step 2: Group NRBs at the same level
            triggers = _group_nrbs_by_level(triggers, weekly_data, tolerance_pct=NRB_LEVEL_TOLERANCE_PCT)
            
            # 🟢 NEW: Ensure ALL triggers have a group_id (even standalone ones)
            triggers = _ensure_all_triggers_have_group_id(triggers)
            
            # Step 3: Find consolidation zones that END with these NRBs
            consolidation_zones = _find_consolidation_zones_with_nrb(
                weekly_data, 
                triggers,
                series_field='close',
                buffer_pct=CONSOLIDATION_BUFFER_PCT,
                min_duration=MIN_CONSOLIDATION_DURATION_WEEKS
            )
            
            # Step 4: Calculate success rates for each zone
            consolidation_zones = _calculate_zone_success_rates(
                base_queryset,
                consolidation_zones,
                series_field='close'
            )
            
            # Step 5: Assign each NRB to its consolidation zone
            triggers = _assign_nrbs_to_zones(triggers, consolidation_zones)

            # Step 6: Extend Group Line & Check Deep Dip
            triggers = _extend_group_lifespan_to_history(
                base_queryset, 
                triggers, 
                'high', 
                tolerance=NRB_LEVEL_TOLERANCE_PCT,
                global_min=global_min,
                global_max=global_max,
                dip_threshold_pct=dip_threshold_pct
            )

            # Step 7: Identify portions of graph close to the Group Level
            triggers = _attach_proximity_zones(base_queryset, triggers, 'close')

            # Step 8: Detect Post-Breakout Whipsaws (PRICE BASED)
            triggers = _detect_post_breakout_whipsaws(
                base_queryset, 
                triggers, 
                value_field='close',
                whipsaw_d1=whipsaw_d1,
                whipsaw_d2=whipsaw_d2
            )

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
                    "rsc_nse": "rsc_nse_ratio",
                    "rsc_nse_ratio": "rsc_nse_ratio",
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
            if total_weeks < nr_weeks + 1:
                return []

            weekly_data = list(
                weekly_qs.values("date", "high", "low", "close", "week")
            )
            
            if not weekly_data:
                return []

            # 1. Get All-Time High/Low for Deep Dip calculation
            stats = param_qs.aggregate(min_val=Min(series_field), max_val=Max(series_field))
            global_min = float(stats['min_val']) if stats['min_val'] else 0
            global_max = float(stats['max_val']) if stats['max_val'] else 0

            # Step 1: Detect NRB breakouts FIRST
            triggers = _detect_narrow_range_break_rolling(weekly_data, nr_weeks, cooldown_weeks)
            
            triggers = _attach_daily_breakout_times_parameter(
                param_qs, series_field, triggers
            )
            
            # Step 2: Group NRBs at the same level
            triggers = _group_nrbs_by_level(triggers, weekly_data, tolerance_pct=NRB_LEVEL_TOLERANCE_PCT)
            
            # 🟢 NEW: Ensure ALL triggers have a group_id (even standalone ones)
            triggers = _ensure_all_triggers_have_group_id(triggers)
            
            # Step 3: Find consolidation zones that END with these NRBs
            consolidation_zones = _find_consolidation_zones_with_nrb(
                weekly_data,
                triggers, 
                series_field='close',
                buffer_pct=CONSOLIDATION_BUFFER_PCT,
                min_duration=MIN_CONSOLIDATION_DURATION_WEEKS
            )
            
            # Step 4: Calculate success rates for each zone
            consolidation_zones = _calculate_zone_success_rates(
                param_qs,
                consolidation_zones,
                series_field=series_field
            )
            
            # Step 5: Assign each NRB to its consolidation zone
            triggers = _assign_nrbs_to_zones(triggers, consolidation_zones)

            # Step 6: Extend Group Line & Check Deep Dip
            triggers = _extend_group_lifespan_to_history(
                param_qs, 
                triggers, 
                series_field, 
                tolerance=NRB_LEVEL_TOLERANCE_PCT,
                global_min=global_min,
                global_max=global_max,
                dip_threshold_pct=dip_threshold_pct
            )

            # Step 7: Identify portions of graph close to the Group Level
            triggers = _attach_proximity_zones(param_qs, triggers, series_field)

            # Step 8: Detect Post-Breakout Whipsaws (PARAMETER)
            triggers = _detect_post_breakout_whipsaws(
                param_qs, 
                triggers, 
                value_field=series_field,
                whipsaw_d1=whipsaw_d1,
                whipsaw_d2=whipsaw_d2
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


def _ensure_all_triggers_have_group_id(triggers):
    """
    🟢 NEW FUNCTION: Ensures EVERY trigger has a group_id.
    For triggers that don't have one (standalone breakouts), create individual groups.
    This ensures ALL blue markers will have cyan duration lines displayed.
    """
    if not triggers:
        return triggers
    
    # Find the highest existing group_id
    max_group_id = 0
    for t in triggers:
        gid = t.get('nrb_group_id')
        if gid and gid > max_group_id:
            max_group_id = gid
    
    # Assign new group_ids to any trigger without one
    next_group_id = max_group_id + 1
    
    for t in triggers:
        if not t.get('nrb_group_id'):
            # Create a standalone group for this trigger
            t['nrb_group_id'] = next_group_id
            t['group_start_time'] = t.get('range_start_time')
            t['group_end_time'] = t.get('range_end_time')
            t['group_level'] = t.get('range_high')
            t['group_nrb_count'] = 1
            next_group_id += 1
    
    print(f"[ENSURE GROUPS] All {len(triggers)} triggers now have group_ids")
    return triggers


# ... (rest of the functions remain exactly the same - I'll include them for completeness)


def _extend_group_lifespan_to_history(daily_qs, triggers, value_field, tolerance=0.05, global_min=0, global_max=0, dip_threshold_pct=0.20):
    """
    1. Extends start time BACKWARDS (History Check).
    2. Stops if Gap > MAX_HISTORY_GAP_DAYS.
    3. CHECKS DEEP DIP: If dip > 50% found, REMOVES GROUP ID (Line gone) but KEEPS TRIGGER (Arrow stays).
    
    🟢 MODIFIED: Made dip check MUCH more lenient (50% fixed threshold instead of dynamic)
    """
    if not triggers:
        return triggers

    print(f"\n[EXTEND HISTORY] Starting with {len(triggers)} triggers")

    # 1. Map groups
    groups = {}
    for t in triggers:
        gid = t.get('nrb_group_id')
        if gid:
            if gid not in groups:
                groups[gid] = {
                    'level': float(t['group_level']),
                    'start_ts': t['group_start_time'],
                    'end_ts': t['group_end_time'],
                    'triggers': []
                }
            groups[gid]['triggers'].append(t)
    
    if not groups:
        return triggers

    print(f"[EXTEND HISTORY] Processing {len(groups)} groups")

    # 2. Get data efficiently
    val_key = 'high' if value_field == 'high' else value_field
    data_points = list(daily_qs.values('trade_date', val_key).order_by('trade_date'))
    
    if not data_points:
        return triggers
        
    processed_data = []
    for d in data_points:
        val = d.get(val_key)
        if val is not None:
            ts = int(datetime.combine(d['trade_date'], datetime.min.time()).timestamp())
            processed_data.append({'ts': ts, 'val': float(val)})

    # Max gap in seconds
    MAX_GAP_SECONDS = MAX_HISTORY_GAP_DAYS * 24 * 60 * 60
    
    # 🟢 FIXED DIP THRESHOLD: 50% drop (very severe) instead of dynamic
    # This ensures only extreme cases remove the cyan line
    SEVERE_DIP_THRESHOLD = 0.50  # 50% drop

    # 3. Extend Backwards & Check Deep Dips
    groups_removed = 0
    for gid, group in groups.items():
        level = group['level']
        upper = level * (1 + tolerance)
        lower = level * (1 - tolerance)
        
        # 🟢 CHANGED: Fixed 50% threshold
        death_line = level * (1 - SEVERE_DIP_THRESHOLD)
        
        current_start_ts = group['start_ts']
        final_start_ts = current_start_ts
        last_valid_touch_ts = current_start_ts 
        
        # --- Extend Left (Backwards) ---
        start_idx = -1
        for i, p in enumerate(processed_data):
            if p['ts'] >= current_start_ts:
                start_idx = i
                break
        
        if start_idx == -1: start_idx = len(processed_data) - 1

        if start_idx > 0:
            for i in range(start_idx - 1, -1, -1):
                p = processed_data[i]
                val = p['val']

                if val > upper: break
                
                if lower <= val <= upper:
                    gap = last_valid_touch_ts - p['ts']
                    if gap > MAX_GAP_SECONDS: break
                    
                    final_start_ts = p['ts']
                    last_valid_touch_ts = p['ts']
        
        # --- DEEP DIP CHECK (Now 50% threshold) ---
        group_end_ts = group['end_ts']
        is_invalid_deep_dip = False
        min_value_in_range = level

        scan_start_idx = -1
        scan_end_idx = -1
        
        for i, p in enumerate(processed_data):
            if scan_start_idx == -1 and p['ts'] >= final_start_ts:
                scan_start_idx = i
            if p['ts'] <= group_end_ts:
                scan_end_idx = i
            else:
                break 
        
        if scan_start_idx != -1 and scan_end_idx != -1:
            for i in range(scan_start_idx, scan_end_idx + 1):
                val = processed_data[i]['val']
                if val < min_value_in_range:
                    min_value_in_range = val
                if val < death_line:
                    is_invalid_deep_dip = True
                    break
        
        if is_invalid_deep_dip:
            # DISSOLVE THE GROUP (Line disappears)
            drop_pct = ((level - min_value_in_range) / level) * 100
            print(f"[EXTEND HISTORY] ❌ Group {gid} removed - {drop_pct:.1f}% dip below level {level:.5f}")
            for t in group['triggers']:
                t['nrb_group_id'] = None
                t['group_start_time'] = None
                t['group_end_time'] = None
                t['group_level'] = None
                t['group_nrb_count'] = None
            groups_removed += 1
        else:
            # Valid group: Update the start time
            extension_weeks = (current_start_ts - final_start_ts) / 604800.0
            print(f"[EXTEND HISTORY] ✅ Group {gid} extended backwards by {extension_weeks:.1f} weeks")
            for t in group['triggers']:
                t['group_start_time'] = final_start_ts

    print(f"[EXTEND HISTORY] Complete: {groups_removed} groups removed, {len(groups) - groups_removed} groups kept")
    return triggers


def _attach_proximity_zones(daily_qs, triggers, value_field, tolerance=NEAR_TOUCH_TOLERANCE_PCT):
    """
    Identifies continuous daily periods where the value is within 'tolerance' % of the group level.
    """
    if not triggers:
        return triggers

    groups = {}
    for t in triggers:
        gid = t.get('nrb_group_id')
        if gid:
            if gid not in groups:
                groups[gid] = {
                    'level': float(t['group_level']),
                    'start_ts': t['group_start_time'],
                    'end_ts': t['group_end_time'],
                    'target_trigger': t 
                }
    
    if not groups:
        return triggers

    data_points = list(daily_qs.values('trade_date', value_field).order_by('trade_date'))
    
    if not data_points:
        return triggers

    processed_data = []
    for d in data_points:
        val = d.get(value_field)
        if val is not None:
            ts = int(datetime.combine(d['trade_date'], datetime.min.time()).timestamp())
            processed_data.append({'ts': ts, 'val': float(val)})

    for gid, group in groups.items():
        level = group['level']
        start_ts = group['start_ts']
        end_ts = group['end_ts']
        
        upper_bound = level * (1 + tolerance)
        lower_bound = level * (1 - tolerance)
        
        touches = []
        current_touch = None
        
        for point in processed_data:
            ts = point['ts']
            val = point['val']
            
            if ts < start_ts:
                continue
            if ts > end_ts:
                break

            is_close = lower_bound <= val <= upper_bound
            
            if is_close:
                diff_pct = abs(val - level) / level * 100
                
                if current_touch is None:
                    current_touch = {
                        'start_time': ts,
                        'end_time': ts,
                        'avg_diff_pct': diff_pct,
                        'min_diff_pct': diff_pct,
                        'max_diff_pct': diff_pct,
                        'count': 1
                    }
                else:
                    current_touch['end_time'] = ts
                    current_touch['count'] += 1
                    current_touch['min_diff_pct'] = min(current_touch['min_diff_pct'], diff_pct)
                    current_touch['max_diff_pct'] = max(current_touch['max_diff_pct'], diff_pct)
                    n = current_touch['count']
                    current_touch['avg_diff_pct'] = (current_touch['avg_diff_pct'] * (n-1) + diff_pct) / n
            else:
                if current_touch:
                    touches.append(current_touch)
                    current_touch = None
        
        if current_touch:
            touches.append(current_touch)
            
        group['target_trigger']['near_touches'] = touches

    return triggers


def _group_nrbs_by_level(triggers, weekly_data, tolerance_pct=0.05, max_gap_bars=5):
    """
    Groups NRBs at same price level.
    """
    if not triggers:
        return triggers
    
    print(f"[NRB GROUPING] Grouping {len(triggers)} NRBs with {tolerance_pct*100}% tolerance")
    
    price_lookup = []
    if weekly_data:
        for row in weekly_data:
            dt = row.get('date')
            if dt:
                ts = int(datetime.combine(dt, datetime.min.time()).timestamp())
                high_val = float(row.get('high', 0))
                price_lookup.append({'time': ts, 'high': high_val})
    
    price_lookup.sort(key=lambda x: x['time'])
    sorted_triggers = sorted(triggers, key=lambda t: t.get('time', 0))
    
    active_groups = []
    closed_groups = []
    group_id = 1
    
    for idx, trigger in enumerate(sorted_triggers):
        range_high = trigger.get('range_high')
        trigger_time = trigger.get('time')
        nrb_id = trigger.get('nrb_id')
        
        if range_high is None or trigger_time is None:
            continue
        
        range_high = float(range_high)
        
        groups_to_close = []
        for group in active_groups:
            gap = idx - group['last_nrb_index']
            if gap > max_gap_bars:
                groups_to_close.append(group)
        
        for group in groups_to_close:
            active_groups.remove(group)
            closed_groups.append(group)
        
        matched_group = None
        
        for group in active_groups:
            group_level = group['level']
            lower_bound = group_level * (1 - tolerance_pct)
            upper_bound = group_level * (1 + tolerance_pct)
            
            if lower_bound <= range_high <= upper_bound:
                last_time = group['end_time']
                current_time = trigger_time
                violation_threshold = upper_bound 
                is_violated = False
                
                for candle in price_lookup:
                    if last_time < candle['time'] < current_time:
                        if candle['high'] > violation_threshold:
                            is_violated = True
                            break
                    if candle['time'] >= current_time:
                        break
                
                if not is_violated:
                    matched_group = group
                    break
        
        if matched_group:
            matched_group['nrb_ids'].append(nrb_id)
            matched_group['end_time'] = trigger_time
            matched_group['last_nrb_index'] = idx
            matched_group['triggers'].append(trigger)
            
            all_levels = [t.get('range_high') for t in matched_group['triggers']]
            matched_group['level'] = sum(all_levels) / len(all_levels)
        else:
            new_group = {
                'id': group_id,
                'level': range_high,
                'nrb_ids': [nrb_id],
                'start_time': trigger_time,
                'end_time': trigger_time,
                'last_nrb_index': idx,
                'triggers': [trigger]
            }
            active_groups.append(new_group)
            group_id += 1
    
    all_groups = closed_groups + active_groups
    
    for group in all_groups:
        for trigger in group['triggers']:
            trigger['nrb_group_id'] = group['id']
            trigger['group_start_time'] = group['start_time']
            trigger['group_end_time'] = group['end_time']
            trigger['group_level'] = group['level']
            trigger['group_nrb_count'] = len(group['nrb_ids'])
    
    return triggers


def _calculate_zone_success_rates(queryset, consolidation_zones, series_field='close'):
    """
    Calculate success rates (3m, 6m, 12m).
    """
    if not consolidation_zones:
        return consolidation_zones
    
    value_lookup = {}
    
    if series_field == 'close':
        for row in queryset.order_by('trade_date'):
            value_lookup[row.trade_date] = float(row.close)
    else:
        for row in queryset.order_by('trade_date'):
            value = getattr(row, series_field, None)
            if value is not None:
                value_lookup[row.trade_date] = float(value)
    
    if not value_lookup:
        return consolidation_zones
    
    all_dates = sorted(value_lookup.keys())
    
    for zone in consolidation_zones:
        zone_end_time = zone.get('end_time')
        if not zone_end_time:
            continue
        
        zone_end_date = datetime.fromtimestamp(zone_end_time).date()
        
        zone_end_value = value_lookup.get(zone_end_date)
        if zone_end_value is None:
            closest_date = min(all_dates, key=lambda d: abs((d - zone_end_date).days), default=None)
            if closest_date:
                zone_end_value = value_lookup[closest_date]
        
        if zone_end_value is None or zone_end_value == 0:
            continue
        
        date_3m = zone_end_date + timedelta(days=90)
        date_6m = zone_end_date + timedelta(days=180)
        date_12m = zone_end_date + timedelta(days=365)
        
        def find_future_value(target_date):
            future_dates = [d for d in all_dates if d >= target_date]
            if not future_dates:
                return None
            return value_lookup.get(future_dates[0])
        
        value_3m = find_future_value(date_3m)
        value_6m = find_future_value(date_6m)
        value_12m = find_future_value(date_12m)
        
        zone['success_rate_3m'] = round(((value_3m / zone_end_value - 1) * 100), 2) if value_3m else None
        zone['success_rate_6m'] = round(((value_6m / zone_end_value - 1) * 100), 2) if value_6m else None
        zone['success_rate_12m'] = round(((value_12m / zone_end_value - 1) * 100), 2) if value_12m else None
    
    return consolidation_zones


def _find_consolidation_zones_with_nrb(weekly_data, nrb_triggers, series_field='close', buffer_pct=0.35, min_duration=4):
    """
    Find consolidation zones.
    """
    if not weekly_data or not nrb_triggers:
        return []
    
    nrb_end_times = set()
    for trigger in nrb_triggers:
        end_time = trigger.get('range_end_time')
        if end_time:
            nrb_end_times.add(end_time)
    
    zones = []
    n = len(weekly_data)
    i = 0
    
    while i < n:
        first_value = weekly_data[i].get(series_field)
        if first_value is None:
            i += 1
            continue
        
        first_value = float(first_value)
        lower_bound = first_value * (1 - buffer_pct)
        upper_bound = first_value * (1 + buffer_pct)
        
        j = i
        values_in_zone = [first_value]
        
        while j < n - 1:
            j += 1
            next_value = weekly_data[j].get(series_field)
            if next_value is None:
                break
            next_value = float(next_value)
            
            if lower_bound <= next_value <= upper_bound:
                values_in_zone.append(next_value)
            else:
                j -= 1
                break
        
        duration = j - i + 1
        
        if duration >= min_duration:
            zone_end_date = weekly_data[j].get("date")
            if zone_end_date:
                zone_end_ts = int(datetime.combine(zone_end_date, datetime.min.time()).timestamp())
                
                if zone_end_ts in nrb_end_times:
                    start_date = weekly_data[i].get("date")
                    start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
                    
                    min_val = min(values_in_zone)
                    max_val = max(values_in_zone)
                    avg_val = sum(values_in_zone) / len(values_in_zone)
                    range_pct = (max_val - min_val) / first_value if first_value > 0 else 0
                    duration_weeks = (zone_end_date - start_date).days / 7.0
                    
                    zones.append({
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
                        'range_pct': range_pct * 100
                    })
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    return zones


def _assign_nrbs_to_zones(triggers, consolidation_zones):
    """
    Assign NRBs to zones.
    """
    if not triggers or not consolidation_zones:
        return triggers
    
    for trigger in triggers:
        nrb_range_start = trigger.get("range_start_time")
        nrb_range_end = trigger.get("range_end_time")
        
        if not nrb_range_start or not nrb_range_end:
            continue
        
        best_zone = None
        best_overlap = 0
        
        for zone in consolidation_zones:
            zone_start = zone['start_time']
            zone_end = zone['end_time']
            
            overlap_start = max(nrb_range_start, zone_start)
            overlap_end = min(nrb_range_end, zone_end)
            
            if overlap_start <= overlap_end:
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > best_overlap:
                    best_overlap = overlap_duration
                    best_zone = zone
        
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
            trigger['zone_success_rate_3m'] = best_zone.get('success_rate_3m')
            trigger['zone_success_rate_6m'] = best_zone.get('success_rate_6m')
            trigger['zone_success_rate_12m'] = best_zone.get('success_rate_12m')
    
    return triggers


def _detect_narrow_range_break_rolling(weekly_data: list, nrb_lookback: int, cooldown_weeks: int = DEFAULT_COOLDOWN_WEEKS):
    """
    Rolling NRB Detection.
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
    last_breakout_idx = None
    
    for i in range(nrb_lookback, n):
        window_start = i - nrb_lookback
        window_end = i
        
        window_weeks = rows[window_start:window_end]
        
        range_high = max(week["high"] for week in window_weeks)
        range_low = min(week["low"] for week in window_weeks)
        
        breakout_week = rows[i]
        
        if breakout_week["high"] > range_high:
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
    return result


def _attach_daily_breakout_times_price(base_queryset, triggers):
    """
    Refine weekly times to daily times.
    """
    if not triggers:
        return triggers

    for t in triggers:
        direction = t.get("direction")
        resistance = t.get("range_high")
        weekly_breakout_ts = t.get("time")

        if weekly_breakout_ts is None or direction != "Bullish Break" or resistance is None:
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
    Refine weekly times to daily times (Parameters).
    """
    if not triggers:
        return triggers

    for t in triggers:
        direction = t.get("direction")
        resistance = t.get("range_high")
        weekly_breakout_ts = t.get("time")

        if weekly_breakout_ts is None or direction != "Bullish Break" or resistance is None:
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
    Bowl detection logic.
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


def _detect_post_breakout_whipsaws(queryset, triggers, value_field='close', whipsaw_d1=None, whipsaw_d2=None):
    """
    V-Shape Whipsaw Logic with DEBUG PRINTS
    """
    if not triggers:
        return triggers

    if not whipsaw_d1 or not whipsaw_d2:
        for t in triggers:
            t['whipsaws'] = []
        return triggers

    print(f"\n[WHIPSAW DEBUG] Started for series: {value_field}")
    print(f"[WHIPSAW DEBUG] D1={whipsaw_d1} weeks, D2={whipsaw_d2} weeks")

    # Convert weeks to seconds for comparison
    WEEK_SECONDS = 7 * 24 * 60 * 60
    d1_seconds = whipsaw_d1 * WEEK_SECONDS
    d2_seconds = whipsaw_d2 * WEEK_SECONDS

    # 1. Fetch data
    earliest_ts = min(t['time'] for t in triggers)
    start_date = datetime.fromtimestamp(earliest_ts).date()

    if value_field == 'close':
        data_points = list(queryset.filter(trade_date__gte=start_date).values('trade_date', 'close').order_by('trade_date'))
    else:
        data_points = list(queryset.filter(trade_date__gte=start_date).values('trade_date', value_field).order_by('trade_date'))

    if not data_points:
        print("[WHIPSAW DEBUG] No data points found after earliest breakout.")
        return triggers

    processed_data = []
    for d in data_points:
        val = d.get('close') if value_field == 'close' else d.get(value_field)
        if val is not None:
            ts = int(datetime.combine(d['trade_date'], datetime.min.time()).timestamp())
            processed_data.append({'ts': ts, 'val': float(val)})

    # 2. Iterate triggers
    for i, trigger in enumerate(triggers):
        trigger['whipsaws'] = []
        breakout_ts = trigger['time']
        
        print(f"\n--- Checking Breakout #{i+1} at TS {breakout_ts} ---")
        
        # Determine Baseline
        start_idx = -1
        baseline_val = None

        for idx, p in enumerate(processed_data):
            if p['ts'] >= breakout_ts:
                baseline_val = p['val']
                start_idx = idx
                break
        
        if start_idx == -1 or baseline_val is None:
            print("[WHIPSAW DEBUG] Could not find baseline value for breakout time.")
            continue
            
        print(f"[WHIPSAW DEBUG] Baseline Value: {baseline_val}")

        # Targets
        drop_target = baseline_val * 0.90   # -10% Drop
        recover_target = baseline_val * 1.05 # +5% Recovery
        
        print(f"[WHIPSAW DEBUG] Targets -> Drop < {drop_target:.4f} | Recover > {recover_target:.4f}")

        # Time Limit for D1
        d1_deadline = breakout_ts + d1_seconds
        
        # --- PHASE 1: Find the Drop (D1) ---
        drop_found = False
        drop_idx = -1
        drop_ts = 0
        min_val_found = baseline_val
        
        for k in range(start_idx, len(processed_data)):
            p = processed_data[k]
            
            if p['ts'] > d1_deadline:
                print(f"[WHIPSAW DEBUG] D1 Timeout. Lowest found: {min_val_found:.4f}")
                break
            
            if p['val'] < min_val_found:
                min_val_found = p['val']

            if p['val'] <= drop_target:
                drop_found = True
                drop_idx = k
                drop_ts = p['ts']
                print(f"[WHIPSAW DEBUG] ✅ DROP FOUND at TS {drop_ts} | Value: {p['val']:.4f}")
                break
        
        # --- PHASE 2: Find the Recovery (D2) ---
        if drop_found:
            d2_deadline = drop_ts + d2_seconds
            recovery_found = False
            
            for k in range(drop_idx + 1, len(processed_data)):
                p = processed_data[k]
                
                if p['ts'] > d2_deadline:
                    print(f"[WHIPSAW DEBUG] D2 Timeout. Recovery failed.")
                    break
                
                if p['val'] >= recover_target:
                    # SUCCESS
                    print(f"[WHIPSAW DEBUG] 🚀 RECOVERY FOUND at TS {p['ts']} | Value: {p['val']:.4f}")
                    trigger['whipsaws'].append({
                        'level': 4,
                        'time': p['ts'],
                        'price': p['val'],
                        'drop_time': drop_ts,
                        'drop_price': processed_data[drop_idx]['val'],
                        'is_v_shape': True
                    })
                    recovery_found = True
                    break
            
            if not recovery_found:
                print("[WHIPSAW DEBUG] Drop found, but no recovery within D2.")
        else:
            print(f"[WHIPSAW DEBUG] No drop found within D1. Lowest was {min_val_found:.4f}")

    return triggers
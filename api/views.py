from datetime import datetime, date, timedelta
from django.core.cache import cache
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Q, Max
from django.utils.timezone import now
from collections import defaultdict

from marketdata.models import Symbol, EodPrice, Parameter, Index, IndexPrice
from core.pattern_recognition import (
    get_pattern_triggers,
    BOWL_MIN_DURATION_DAYS,
    NRB_LOOKBACK,
    DEFAULT_COOLDOWN_WEEKS,
)

from .serializers import SymbolListItemSerializer
from .pagination import SymbolPagination
from .utils import relevance


def _calculate_total_nrb_duration(triggers):
    """
    Build ALL DISTINCT contiguous NRB groups (non-overlapping).
    
    For EACH group:
      - start_date
      - end_date
      - duration in weeks
    
    Attach per-trigger:
      - nrb_group_id
      - group_start_time
      - group_end_time
      - group_duration_weeks
    """
    
    if not triggers:
        return triggers
    
    # Sort by breakout time
    sorted_triggers = sorted(triggers, key=lambda t: t["time"])
    
    def are_contiguous(nrb1, nrb2):
        """Check if two NRBs are part of same contiguous group"""
        r1 = nrb1.get("range_high")
        r2 = nrb2.get("range_high")
        
        if r1 is None or r2 is None:
            return False
        
        # Allow 20% tolerance in resistance levels
        buffer = 0.20 * abs(r1)
        return abs(r2 - r1) <= buffer
    
    # Build contiguous groups
    contiguous_groups = []
    current_group = [sorted_triggers[0]]
    
    for i in range(1, len(sorted_triggers)):
        if are_contiguous(sorted_triggers[i - 1], sorted_triggers[i]):
            current_group.append(sorted_triggers[i])
        else:
            # Save current group and start new one
            contiguous_groups.append(current_group)
            current_group = [sorted_triggers[i]]
    
    # Don't forget the last group
    contiguous_groups.append(current_group)
    
    # Assign group metadata to each trigger
    for group_id, group in enumerate(contiguous_groups, start=1):
        start_ts = group[0].get("range_start_time")
        end_ts = group[-1].get("range_end_time")
        
        if start_ts and end_ts:
            start_date = datetime.fromtimestamp(start_ts).date()
            end_date = datetime.fromtimestamp(end_ts).date()
            duration_weeks = (end_date - start_date).days / 7.0  # Use float for precision
        else:
            duration_weeks = 0
        
        for t in group:
            t["nrb_group_id"] = group_id
            t["group_start_time"] = start_ts
            t["group_end_time"] = end_ts
            t["group_duration_weeks"] = duration_weeks
    
    print(f"[NRB DEBUG] Total NRB groups found: {len(contiguous_groups)}")
    for i, g in enumerate(contiguous_groups, 1):
        print(f"  Group {i}: {len(g)} NRBs, Duration: {g[0].get('group_duration_weeks'):.2f} weeks")
    
    return triggers


class SymbolListView(APIView):
    """
    Unified symbol + index search with pagination + sector info.
    """

    pagination_class = SymbolPagination

    def get(self, request, *args, **kwargs):
        query = request.query_params.get("q", "").strip()

        # ================================
        # 1. Load Symbols (with sector)
        # ================================
        symbol_qs = (
            Symbol.objects
            .filter(eodprice__isnull=False)
            .distinct()
        )

        if query:
            symbol_qs = symbol_qs.filter(
                Q(symbol__icontains=query) |
                Q(company_name__icontains=query) |
                Q(sector__name__icontains=query)
            )

        symbol_qs = symbol_qs.values(
            "id",
            "symbol",
            "company_name",
            "sector__name",
            "sector_id",
        )

        symbol_list = [
            {
                "id": s["id"],
                "symbol": s["symbol"],
                "name": s["company_name"] or s["symbol"],
                "sector": s["sector__name"],
                "sector_id": s["sector_id"],
                "type": "symbol",
            }
            for s in symbol_qs
        ]

        # ================================
        # 2. Load Indices
        # ================================
        index_qs = (
            Index.objects
            .filter(indexprice__isnull=False)
            .distinct()
        )

        if query:
            index_qs = index_qs.filter(
                Q(symbol__icontains=query) |
                Q(name__icontains=query)
            )

        index_qs = index_qs.values("id", "symbol", "name")

        index_list = [
            {
                "id": idx["id"],
                "symbol": idx["symbol"],
                "name": idx["name"],
                "sector": None,
                "sector_id": None,
                "type": "index",
            }
            for idx in index_qs
        ]

        # ================================
        # 3. Merge & Sort (using relevance)
        # ================================
        combined = symbol_list + index_list
        combined = sorted(combined, key=lambda x: relevance(x, query))

        # ================================
        # 4. Pagination
        # ================================
        paginator = self.pagination_class()
        paginated_data = paginator.paginate_queryset(combined, request)

        serializer = SymbolListItemSerializer(paginated_data, many=True)
        return paginator.get_paginated_response(serializer.data)


class PatternScanView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            scrip = request.query_params.get("scrip")
            pattern = request.query_params.get("pattern")

            nrb_lookback = NRB_LOOKBACK

            success_rate_raw = request.query_params.get("success_rate", "0")
            success_rate = float(success_rate_raw) if success_rate_raw != "" else 0.0

            weeks_param = request.query_params.get("weeks")
            weeks = int(weeks_param) if weeks_param is not None else None

            series_param = request.query_params.get("series")
            series = series_param.strip().lower() if series_param else None

            # Get cooldown_weeks from frontend, default to DEFAULT_COOLDOWN_WEEKS
            cooldown_weeks_param = request.query_params.get("cooldown_weeks")
            cooldown_weeks = int(cooldown_weeks_param) if cooldown_weeks_param else DEFAULT_COOLDOWN_WEEKS

            if not scrip or not pattern:
                print("Scrip and Pattern are required.", scrip, pattern)
                return Response(
                    {"error": "Scrip and Pattern are required."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        except ValueError:
            return Response(
                {"error": "Invalid numerical input for success_rate, weeks, or cooldown_weeks."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        total_rows = EodPrice.objects.filter(symbol__symbol=scrip).count()
        ema_rows = Parameter.objects.filter(
            symbol__symbol=scrip, ema50__isnull=False
        ).count()

        # Get pattern triggers
        trigger_markers = get_pattern_triggers(
            scrip=scrip,
            pattern=pattern,
            nrb_lookback=nrb_lookback,
            success_rate=success_rate,
            weeks=weeks,
            series=series,
            cooldown_weeks=cooldown_weeks,
        )

        # ✅ FIX: Ensure NRB grouping is calculated for NRB-related patterns
        if trigger_markers and pattern in ["NRB", "NR4", "NR7"]:
            trigger_markers = _calculate_total_nrb_duration(trigger_markers)

        ohlcv_qs = EodPrice.objects.filter(symbol__symbol=scrip).order_by("trade_date")
        ohlcv_data = [
            {
                "time": int(
                    datetime.combine(row.trade_date, datetime.min.time()).timestamp()
                ),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
            }
            for row in ohlcv_qs
        ]

        # ----- series_data for Parameter-based filters -----
        series_data = []
        series_data_ema5 = []
        series_data_ema10 = []
        
        valid_series_fields = {
            "ema21": "ema21",
            "ema50": "ema50",
            "ema200": "ema200",
            "rsc30": "rsc_sensex_ratio",
            "rsc500": "rsc500",
        }

        if series in valid_series_fields:
            field_name = valid_series_fields[series]
            
            if series == "rsc30":
                param_qs = (
                    Parameter.objects.filter(symbol__symbol=scrip)
                    .exclude(rsc_sensex_ratio__isnull=True)
                    .order_by("trade_date")
                )

                # GRAY LINE - Raw ratio
                series_data = [
                    {
                        "time": int(
                            datetime.combine(
                                row.trade_date, datetime.min.time()
                            ).timestamp()
                        ),
                        "value": float(row.rsc_sensex_ratio),
                    }
                    for row in param_qs
                    if row.rsc_sensex_ratio is not None
                ]

                # RED LINE - EMA5
                series_data_ema5 = [
                    {
                        "time": int(
                            datetime.combine(
                                row.trade_date, datetime.min.time()
                            ).timestamp()
                        ),
                        "value": float(row.rsc_sensex_ema5),
                    }
                    for row in param_qs
                    if row.rsc_sensex_ema5 is not None
                ]

                # BLUE LINE - EMA10
                series_data_ema10 = [
                    {
                        "time": int(
                            datetime.combine(
                                row.trade_date, datetime.min.time()
                            ).timestamp()
                        ),
                        "value": float(row.rsc_sensex_ema10),
                    }
                    for row in param_qs
                    if row.rsc_sensex_ema10 is not None
                ]
            else:
                param_qs = (
                    Parameter.objects.filter(symbol__symbol=scrip)
                    .exclude(**{f"{field_name}__isnull": True})
                    .order_by("trade_date")
                )

                series_data = [
                    {
                        "time": int(
                            datetime.combine(
                                row.trade_date, datetime.min.time()
                            ).timestamp()
                        ),
                        "value": float(getattr(row, field_name)),
                    }
                    for row in param_qs
                ]

        # ✅ FIX: Build RSC lookup BEFORE processing markers
        rsc_lookup = {}
        if series == "rsc30" and series_data:
            for point in series_data:
                rsc_lookup[point["time"]] = point["value"]

        # ✅ FIX: Extract unique NRB groups using proper grouping
        nrb_groups = []
        if trigger_markers:
            groups_dict = defaultdict(list)
            
            for marker in trigger_markers:
                group_id = marker.get("nrb_group_id")
                if group_id:
                    groups_dict[group_id].append(marker)
            
            # Build nrb_groups array with proper metadata
            for group_id in sorted(groups_dict.keys()):
                group = groups_dict[group_id]
                first_marker = group[0]
                
                # Calculate average range_high for the group
                valid_range_highs = [m.get("range_high") for m in group if m.get("range_high") is not None]
                avg_range_high = sum(valid_range_highs) / len(valid_range_highs) if valid_range_highs else None
                
                nrb_groups.append({
                    "group_id": group_id,
                    "duration_weeks": first_marker.get("group_duration_weeks", 0),
                    "start_time": first_marker.get("group_start_time"),
                    "end_time": first_marker.get("group_end_time"),
                    "num_nrbs": len(group),
                    "avg_range_high": avg_range_high,
                })

        # Debug log
        print(f"[VIEW DEBUG] Extracted {len(nrb_groups)} NRB groups")
        for g in nrb_groups:
            duration = g.get('duration_weeks', 0)
            print(f"  Group {g['group_id']}: {g['num_nrbs']} NRBs, {duration:.2f} weeks")

        # ✅ FIX: Process markers with RSC conversion done once per marker
        markers = []
        for row in trigger_markers:
            score = row.get("score", 0.0)
            pattern_id = row.get("pattern_id")

            if pattern == "Bowl" and pattern_id is not None:
                text = f"Bowl Pattern #{pattern_id} | Score: {score:.2f}"
            else:
                text = f"Pattern: {pattern} | Success Score: {score:.2f}"

            # Get original range values
            range_low = row.get("range_low")
            range_high = row.get("range_high")
            
            # ✅ FIX: Convert to RSC scale if needed (make copies to avoid mutation)
            if series == "rsc30" and range_low is not None and range_high is not None:
                range_start_time = row.get("range_start_time")
                range_end_time = row.get("range_end_time")
                
                if range_start_time and range_end_time:
                    # Get RSC values during the range period
                    rsc_values_in_range = [
                        v for t, v in rsc_lookup.items() 
                        if range_start_time <= t <= range_end_time
                    ]
                    
                    if rsc_values_in_range:
                        # Use min/max RSC in range
                        range_low = min(rsc_values_in_range)
                        range_high = max(rsc_values_in_range)

            markers.append({
                "time": row["time"],
                "position": "aboveBar",
                "color": "#2196F3",
                "shape": "circle",
                "text": text,
                "pattern_id": pattern_id,
                "range_low": range_low,
                "range_high": range_high,
                "range_start_time": row.get("range_start_time"),
                "range_end_time": row.get("range_end_time"),
                "nrb_id": row.get("nrb_id"),
                "nrb_group_id": row.get("nrb_group_id"),  # Include group ID in marker
                "nr_high": row.get("nr_high"),
                "nr_low": row.get("nr_low"),
                "direction": row.get("direction"),
            })

        # ✅ FIX: Calculate total duration properly with safe access
        total_nrb_duration_weeks = sum(
            g.get("duration_weeks", 0) for g in nrb_groups
        ) if nrb_groups else 0

        response_data = {
            "scrip": scrip,
            "pattern": pattern,
            "price_data": ohlcv_data,
            "markers": markers,
            "series": series,
            "series_data": series_data,
            "series_data_ema5": series_data_ema5,
            "series_data_ema10": series_data_ema10,
            "total_nrb_duration_weeks": round(total_nrb_duration_weeks, 2),
            "nrb_groups": nrb_groups,
            "debug": {
                "total_rows": total_rows,
                "ema_rows": ema_rows,
                "triggers_found": len(trigger_markers),
                "markers_created": len(markers),
                "success_rate_filter": success_rate,
                "weeks_param": weeks,
                "cooldown_weeks": cooldown_weeks,
                "bowl_min_duration_days": BOWL_MIN_DURATION_DAYS,
                "nrb_default_lookback": NRB_LOOKBACK,
                "series_param": series,
                "series_data_points": len(series_data),
                "series_data_ema5_points": len(series_data_ema5),
                "series_data_ema10_points": len(series_data_ema10),
                "nrb_groups_count": len(nrb_groups),
            },
        }
        
        # Debug: Print what we're about to send
        print(f"[VIEW DEBUG] Response data keys: {response_data.keys()}")
        print(f"[VIEW DEBUG] Response total_nrb_duration_weeks: {response_data.get('total_nrb_duration_weeks')}")
        print(f"[VIEW DEBUG] Response nrb_groups count: {len(nrb_groups)}")

        return Response(response_data, status=status.HTTP_200_OK)


class PriceHistoryView(APIView):
    """
    Fetch historical OHLC price data for a given scrip (Symbol or Index).
    """

    CACHE_TIMEOUT = 60 * 60 * 24   # 24 hours

    def get(self, request, *args, **kwargs):

        # ================================
        # Validate Inputs
        # ================================
        scrip = request.query_params.get("scrip")
        years_raw = request.query_params.get("years", 10)

        if not scrip:
            return Response(
                {"error": "Query parameter 'scrip' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            years = int(years_raw)
            if years <= 0:
                raise ValueError
        except:
            return Response(
                {"error": "'years' must be a positive integer."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        cutoff_date = date.today() - timedelta(days=years * 365)

        # ================================
        # Caching Layer
        # ================================
        cache_key = f"price-history:{scrip}:{years}"
        cached = cache.get(cache_key)
        if cached:
            return Response(cached, status=status.HTTP_200_OK)

        # ================================
        # Determine if scrip is a Symbol or Index
        # ================================
        symbol_obj = Symbol.objects.filter(symbol=scrip).first()
        index_obj = Index.objects.filter(symbol=scrip).first()

        if not symbol_obj and not index_obj:
            return Response(
                {"error": f"No stock or index found with symbol '{scrip}'."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # ================================
        # Query the correct price table
        # ================================
        if symbol_obj:
            price_qs = (
                EodPrice.objects
                .filter(symbol=symbol_obj, trade_date__gte=cutoff_date)
                .order_by("trade_date")
                .values("trade_date", "open", "high", "low", "close")
            )
        else:
            price_qs = (
                IndexPrice.objects
                .filter(index=index_obj, trade_date__gte=cutoff_date)
                .order_by("trade_date")
                .values("trade_date", "open", "high", "low", "close")
            )

        if not price_qs.exists():
            return Response(
                {"error": "No price data for the given scrip in the selected date range."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # ================================
        # Serialize data
        # ================================
        price_data = [
            {
                "time": int(datetime.combine(row["trade_date"], datetime.min.time()).timestamp()),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
            }
            for row in price_qs
        ]

        response = {
            "scrip": scrip,
            "price_data": price_data,
            "records": len(price_data),
        }

        # Store in cache
        cache.set(cache_key, response, timeout=self.CACHE_TIMEOUT)

        return Response(response, status=status.HTTP_200_OK)


class Week52HighView(APIView):
    """
    Returns the 52-week high for a scrip.
    Supports both Symbol (stocks) and Index (Sensex, Nifty500).
    """

    def get(self, request, *args, **kwargs):
        # -----------------------------------------
        # 1. Validate Input
        # -----------------------------------------
        scrip = request.query_params.get("scrip")
        if not scrip:
            return Response(
                {"error": "Query parameter 'scrip' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # -----------------------------------------
        # 2. Compute cutoff date (52 weeks)
        # -----------------------------------------
        cutoff_date = date.today() - timedelta(days=365)

        # -----------------------------------------
        # 3. Check if scrip is a STOCK
        # -----------------------------------------
        stock_exists = Symbol.objects.filter(symbol=scrip).exists()

        if stock_exists:
            data = (
                EodPrice.objects.filter(
                    symbol__symbol=scrip,
                    trade_date__gte=cutoff_date
                )
                .aggregate(week52_high=Max("high"))
            )
        else:
            # -----------------------------------------
            # 4. Check if scrip is an INDEX
            # -----------------------------------------
            index_exists = Index.objects.filter(symbol=scrip).exists()

            if not index_exists:
                return Response(
                    {
                        "error": f"'{scrip}' is not a valid symbol or index.",
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            data = (
                IndexPrice.objects.filter(
                    index__symbol=scrip,
                    trade_date__gte=cutoff_date
                )
                .aggregate(week52_high=Max("high"))
            )

        # -----------------------------------------
        # 5. Extract result
        # -----------------------------------------
        week52_high = data.get("week52_high")

        if week52_high is None:
            return Response(
                {
                    "scrip": scrip,
                    "52week_high": None,
                    "message": "No price data found for the past 52 weeks.",
                },
                status=status.HTTP_200_OK,
            )

        # -----------------------------------------
        # 6. Return Response
        # -----------------------------------------
        return Response(
            {
                "scrip": scrip,
                "52week_high": float(week52_high),
                "cutoff_date": cutoff_date.isoformat(),
            },
            status=status.HTTP_200_OK,
        )
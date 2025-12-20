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
    CONSOLIDATION_BUFFER_PCT,
    MIN_CONSOLIDATION_DURATION_WEEKS,
    _find_consolidation_zones,
)

from .serializers import SymbolListItemSerializer
from .pagination import SymbolPagination
from .utils import relevance


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

        # Get pattern triggers with consolidation zones
        trigger_markers = get_pattern_triggers(
            scrip=scrip,
            pattern=pattern,
            nrb_lookback=nrb_lookback,
            success_rate=success_rate,
            weeks=weeks,
            series=series,
            cooldown_weeks=cooldown_weeks,
        )

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

        # Get weekly data for consolidation zone calculation
        weekly_data = []
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

                # Get weekly data for zone calculation
                from django.db.models.functions import TruncWeek
                from django.db.models import Min, Max
                weekly_qs = (
                    param_qs
                    .annotate(week=TruncWeek("trade_date"))
                    .values("week")
                    .annotate(
                        high=Max(field_name),
                        low=Min(field_name),
                        close=Max(field_name),
                        date=Max("trade_date"),
                    )
                    .order_by("week")
                )
                weekly_data = list(weekly_qs.values("date", "high", "low", "close", "week"))

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

                # Get weekly data for zone calculation
                from django.db.models.functions import TruncWeek
                from django.db.models import Min, Max
                weekly_qs = (
                    param_qs
                    .annotate(week=TruncWeek("trade_date"))
                    .values("week")
                    .annotate(
                        high=Max(field_name),
                        low=Min(field_name),
                        close=Max(field_name),
                        date=Max("trade_date"),
                    )
                    .order_by("week")
                )
                weekly_data = list(weekly_qs.values("date", "high", "low", "close", "week"))

        # Get consolidation zones separately for the response
        consolidation_zones = []
        if weekly_data and pattern == "Narrow Range Break":
            from core.pattern_recognition import CONSOLIDATION_BUFFER_PCT, MIN_CONSOLIDATION_DURATION_WEEKS
            consolidation_zones = _find_consolidation_zones(
                weekly_data,
                series_field='close',
                buffer_pct=CONSOLIDATION_BUFFER_PCT,
                min_duration=MIN_CONSOLIDATION_DURATION_WEEKS
            )

        # Build RSC lookup for marker conversion
        rsc_lookup = {}
        if series == "rsc30" and series_data:
            for point in series_data:
                rsc_lookup[point["time"]] = point["value"]

        # Extract unique consolidation zones from markers
        zones_by_id = {}
        if trigger_markers:
            for marker in trigger_markers:
                zone_id = marker.get("consolidation_zone_id")
                if zone_id and zone_id not in zones_by_id:
                    zones_by_id[zone_id] = {
                        "zone_id": zone_id,
                        "start_time": marker.get("zone_start_time"),
                        "end_time": marker.get("zone_end_time"),
                        "duration_weeks": marker.get("zone_duration_weeks", 0),
                        "min_value": marker.get("zone_min_value"),
                        "max_value": marker.get("zone_max_value"),
                        "avg_value": marker.get("zone_avg_value"),
                        "range_pct": marker.get("zone_range_pct", 0),
                        "num_nrbs": 0,
                    }
                
                if zone_id:
                    zones_by_id[zone_id]["num_nrbs"] += 1

        consolidation_groups = list(zones_by_id.values())

        # Debug log
        print(f"[VIEW DEBUG] Extracted {len(consolidation_groups)} consolidation zones")
        for g in consolidation_groups:
            print(f"  Zone {g['zone_id']}: {g['num_nrbs']} NRBs, {g['duration_weeks']:.2f} weeks, Range: {g['range_pct']:.1f}%")

        # Process markers
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
            
            # Convert to RSC scale if needed
            if series == "rsc30" and range_low is not None and range_high is not None:
                range_start_time = row.get("range_start_time")
                range_end_time = row.get("range_end_time")
                
                if range_start_time and range_end_time:
                    rsc_values_in_range = [
                        v for t, v in rsc_lookup.items() 
                        if range_start_time <= t <= range_end_time
                    ]
                    
                    if rsc_values_in_range:
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
                "consolidation_zone_id": row.get("consolidation_zone_id"),
                "nr_high": row.get("nr_high"),
                "nr_low": row.get("nr_low"),
                "direction": row.get("direction"),
            })

        # Calculate total duration
        total_consolidation_duration_weeks = sum(
            z.get("duration_weeks", 0) for z in consolidation_groups
        ) if consolidation_groups else 0

        response_data = {
            "scrip": scrip,
            "pattern": pattern,
            "price_data": ohlcv_data,
            "markers": markers,
            "series": series,
            "series_data": series_data,
            "series_data_ema5": series_data_ema5,
            "series_data_ema10": series_data_ema10,
            "total_consolidation_duration_weeks": round(total_consolidation_duration_weeks, 2),
            "consolidation_zones": consolidation_groups,  # Changed from nrb_groups
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
                "consolidation_zones_count": len(consolidation_groups),
            },
        }
        
        print(f"[VIEW DEBUG] Response data keys: {response_data.keys()}")
        print(f"[VIEW DEBUG] Response total_consolidation_duration_weeks: {response_data.get('total_consolidation_duration_weeks')}")
        print(f"[VIEW DEBUG] Response consolidation_zones count: {len(consolidation_groups)}")

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

        cache.set(cache_key, response, timeout=self.CACHE_TIMEOUT)

        return Response(response, status=status.HTTP_200_OK)


class Week52HighView(APIView):
    """
    Returns the 52-week high for a scrip.
    Supports both Symbol (stocks) and Index (Sensex, Nifty500).
    """

    def get(self, request, *args, **kwargs):
        scrip = request.query_params.get("scrip")
        if not scrip:
            return Response(
                {"error": "Query parameter 'scrip' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        cutoff_date = date.today() - timedelta(days=365)

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

        return Response(
            {
                "scrip": scrip,
                "52week_high": float(week52_high),
                "cutoff_date": cutoff_date.isoformat(),
            },
            status=status.HTTP_200_OK,
        )
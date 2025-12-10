from datetime import datetime, date, timedelta

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Q
from marketdata.models import Symbol, EodPrice, Parameter
from api.serializers import SymbolSerializer
from core.pattern_recognition import (
    get_pattern_triggers,
    BOWL_MIN_DURATION_DAYS,
    NRB_LOOKBACK,
)


class SymbolListView(APIView):
    def get(self, request, *args, **kwargs):
        query = request.query_params.get("q")

        qs = Symbol.objects.all()

        if query:
            qs = qs.filter(
                Q(symbol__icontains=query) | Q(company_name__icontains=query)
            )

        qs = qs.order_by("symbol")

        serializer = SymbolSerializer(qs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


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

            if not scrip or not pattern:
                return Response(
                    {"error": "Scrip and Pattern are required."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        except ValueError:
            return Response(
                {"error": "Invalid numerical input for success_rate or weeks."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ✅ use symbol__symbol; no EMA_50 on EodPrice now
        total_rows = EodPrice.objects.filter(symbol__symbol=scrip).count()
        ema_rows = Parameter.objects.filter(
            symbol__symbol=scrip, ema50__isnull=False
        ).count()

        trigger_markers = get_pattern_triggers(
            scrip=scrip,
            pattern=pattern,
            nrb_lookback=nrb_lookback,
            success_rate=success_rate,
            weeks=weeks,
            series=series,
        )

        # ✅ use symbol__symbol and trade_date
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
        valid_series_fields = {
            "ema21": "ema21",
            "ema50": "ema50",
            "ema200": "ema200",
            "rsc30": "rsc30",
            "rsc500": "rsc500",
        }

        if series in valid_series_fields:
            field_name = valid_series_fields[series]
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
                    "value": getattr(row, field_name),
                }
                for row in param_qs
            ]

        # ----- markers -----
        markers = []
        for row in trigger_markers:
            score = row.get("score", 0.0)
            pattern_id = row.get("pattern_id")

            if pattern == "Bowl" and pattern_id is not None:
                text = f"Bowl Pattern #{pattern_id} | Score: {score:.2f}"
            else:
                text = f"Pattern: {pattern} | Success Score: {score:.2f}"

            markers.append(
                {
                    "time": row["time"],
                    "position": "aboveBar",
                    "color": "#2196F3",
                    "shape": "circle",
                    "text": text,
                    "pattern_id": pattern_id,
                    "range_low": row.get("range_low"),
                    "range_high": row.get("range_high"),
                    "range_start_time": row.get("range_start_time"),
                    "range_end_time": row.get("range_end_time"),
                    "nrb_id": row.get("nrb_id"),
                    "nr_high": row.get("nr_high"),
                    "nr_low": row.get("nr_low"),
                    "direction": row.get("direction"),
                }
            )

        response_data = {
            "scrip": scrip,
            "pattern": pattern,
            "price_data": ohlcv_data,
            "markers": markers,
            "series": series,
            "series_data": series_data,
            "debug": {
                "total_rows": total_rows,
                "ema_rows": ema_rows,
                "triggers_found": len(trigger_markers),
                "markers_created": len(markers),
                "success_rate_filter": success_rate,
                "weeks_param": weeks,
                "bowl_min_duration_days": BOWL_MIN_DURATION_DAYS,
                "nrb_default_lookback": NRB_LOOKBACK,
                "series_param": series,
                "series_data_points": len(series_data),
            },
        }

        return Response(response_data, status=status.HTTP_200_OK)


class PriceHistoryView(APIView):
    def get(self, request, *args, **kwargs):
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
        except (TypeError, ValueError):
            return Response(
                {"error": "Query parameter 'years' must be a positive integer."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        cutoff_date = date.today() - timedelta(days=years * 365)

        # ✅ use symbol__symbol and trade_date
        price_queryset = (
            EodPrice.objects.filter(
                symbol__symbol=scrip, trade_date__gte=cutoff_date
            )
            .order_by("trade_date")
        )

        price_data = [
            {
                "time": int(
                    datetime.combine(row.trade_date, datetime.min.time()).timestamp()
                ),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
            }
            for row in price_queryset
        ]

        return Response(
            {
                "scrip": scrip,
                "price_data": price_data,
                "records": len(price_data),
            },
            status=status.HTTP_200_OK,
        )

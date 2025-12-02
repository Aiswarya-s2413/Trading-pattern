from datetime import datetime, date, timedelta

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from core.pattern_recognition import get_pattern_triggers, BOWL_MIN_DURATION_DAYS
from api.models import PriceFeature


class PatternScanView(APIView):
    """
    API endpoint to fetch price data and apply complex pattern recognition filters.
    """

    def get(self, request, *args, **kwargs):
        try:
            scrip = request.query_params.get("scrip")
            pattern = request.query_params.get("pattern")
            nrb_lookback = int(request.query_params.get("nrb_lookback", 7))
            success_rate = float(request.query_params.get("success_rate", 0.0))

            if not scrip or not pattern:
                return Response(
                    {"error": "Scrip and Pattern are required."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        except ValueError:
            return Response(
                {"error": "Invalid numerical input for lookback or success_rate."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if pattern == "Bowl" and nrb_lookback < BOWL_MIN_DURATION_DAYS:
            return Response(
                {
                    "error": f"Pattern Incompatibility: Bowl requires at least {BOWL_MIN_DURATION_DAYS} days of data. Use the Weekly TimeFrame for this pattern."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Check data availability for debugging
        total_rows = PriceFeature.objects.filter(symbol=scrip).count()
        ema_rows = PriceFeature.objects.filter(
            symbol=scrip, EMA_50__isnull=False
        ).count()

        # Core logic
        trigger_markers = get_pattern_triggers(
            scrip=scrip,
            pattern=pattern,
            nrb_lookback=nrb_lookback,
            success_rate=success_rate,
        )

        # OHLCV data
        ohlcv_qs = PriceFeature.objects.filter(symbol=scrip).order_by("date")
        ohlcv_data = [
            {
                "time": int(row.date.strftime("%s")),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
            }
            for row in ohlcv_qs
        ]

        # Transform triggers into markers format
        markers = []
        for row in trigger_markers:
            score = row.get("score", 0.0)
            pattern_id = row.get("pattern_id")

            # Determine marker text based on pattern type
            if pattern == "Bowl" and pattern_id is not None:
                # For bowl patterns, we have 3 markers per pattern (left rim, bottom, right rim)
                # Use pattern_id to distinguish different bowls
                text = f"Bowl Pattern #{pattern_id} | Score: {score:.2f}"
            else:
                text = f"Pattern: {pattern} | Success Score: {score:.2f}"

            markers.append(
                {
                    "time": row["time"],
                    "position": "belowBar",
                    "color": "#2196F3",
                    "shape": "circle",
                    "text": text,
                    "pattern_id": pattern_id,  # None for NRB, int for Bowl
                }
            )

        response_data = {
            "scrip": scrip,
            "pattern": pattern,
            "price_data": ohlcv_data,
            "markers": markers,
            # Debug info (remove in production if needed)
            "debug": {
                "total_rows": total_rows,
                "ema_rows": ema_rows,
                "triggers_found": len(trigger_markers),
                "markers_created": len(markers),
                "success_rate_filter": success_rate,
            },
        }

        return Response(response_data, status=status.HTTP_200_OK)


class PriceHistoryView(APIView):
    """
    Returns raw OHLC data for the requested scrip. Primarily used to hydrate the
    initial chart view before any filters are applied on the frontend.
    """

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
        price_queryset = PriceFeature.objects.filter(
            symbol=scrip, date__gte=cutoff_date
        ).order_by("date")

        price_data = [
            {
                "time": int(
                    datetime.combine(row.date, datetime.min.time()).timestamp()
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

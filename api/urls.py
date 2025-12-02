# api/urls.py
from django.urls import path
from .views import PatternScanView, PriceHistoryView

urlpatterns = [
    path("pattern-scan/", PatternScanView.as_view(), name="pattern-scan"),
    path("price-history/", PriceHistoryView.as_view(), name="price-history"),
]

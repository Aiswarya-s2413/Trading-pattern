from django.urls import path
from .views import PatternScanView, PriceHistoryView, SymbolListView

urlpatterns = [
    path("pattern-scan/", PatternScanView.as_view(), name="pattern-scan"),
    path("price-history/", PriceHistoryView.as_view(), name="price-history"),
    path("symbols/", SymbolListView.as_view(), name="symbol-list"),
]

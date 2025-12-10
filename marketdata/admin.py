from django.contrib import admin
from .models import Sectors, Symbol, EodPrice, Index, IndexPrice, Parameter

@admin.register(Symbol)
class SymbolAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "symbol",
        "company_name",
        "sector",
        "market_type",
    )

    search_fields = ("symbol", "company_name")
    list_filter = ("sector", "market_type")


@admin.register(EodPrice)
class EodPriceAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    # Enable search on symbol name and date
    search_fields = (
        "symbol__symbol",   # symbol field of Symbol model
        "trade_date",
    )

    # Add filtering options
    list_filter = (
        "symbol",
        "trade_date",
    )

    # Improves performance in admin by joining related tables
    list_select_related = ("symbol",)

    # Optional: Order results newest-first
    ordering = ("-trade_date",)


@admin.register(Parameter)
class ParameterAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "symbol",
        "trade_date",
        "rsc30",
        "rsc500",
        "ema21",
        "ema50",
        "ema200",
    )

    # Enable search on symbol name and date
    search_fields = (
        "symbol__symbol",   # symbol field of Symbol model
        "trade_date",
    )

    # Add filtering options
    list_filter = (
        "symbol",
        "trade_date",
    )

    # Improves performance in admin by joining related tables
    list_select_related = ("symbol",)

    # Optional: Order results newest-first
    ordering = ("-trade_date",)

admin.site.register(Sectors)
admin.site.register(Index)
admin.site.register(IndexPrice)
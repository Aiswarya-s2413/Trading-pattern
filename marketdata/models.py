from django.db import models
from django.core.exceptions import ValidationError

# Create your models here.
class Sectors(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name = "Sector"
        verbose_name_plural = "Sectors"
        ordering = ['name']


def validate_market_type(value):
    if value not in ['NSE', 'BSE']:
        raise ValidationError('Invalid market type')


class Symbol(models.Model):
    symbol = models.CharField(max_length=20, unique=True) 
    company_name = models.CharField(max_length=255, null=True, blank=True)
    sector = models.ForeignKey(Sectors, on_delete=models.SET_NULL, null=True)
    market_type = models.CharField(max_length=20, null=True, blank=True, validators=[validate_market_type])

    def __str__(self):
        return self.symbol
    
    class Meta:
        verbose_name = "Symbol"
        verbose_name_plural = "Symbols"
        ordering = ['symbol']

class EodPrice(models.Model):
    trade_date = models.DateField()
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE)
    open = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    high = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    low = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    close = models.DecimalField(max_digits=12, decimal_places=2)
    volume = models.BigIntegerField()
    market_cap = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)

    class Meta:
        unique_together = ("trade_date", "symbol")
        verbose_name = "EOD Price"
        verbose_name_plural = "EOD Prices"
        ordering = ['trade_date']
        indexes = [
            models.Index(fields=['symbol', 'trade_date']),
            models.Index(fields=['trade_date']),
        ]

    def __str__(self):
        return f"{self.symbol} - {self.trade_date}"


class Parameter(models.Model):
    trade_date = models.DateField()
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE)

    closing_price = models.DecimalField(max_digits=12, decimal_places=2)
    rsc30 = models.DecimalField(max_digits=12, decimal_places=4, null=True)
    rsc500 = models.DecimalField(max_digits=12, decimal_places=4, null=True)

    ema21 = models.DecimalField(max_digits=12, decimal_places=4, null=True)
    ema50 = models.DecimalField(max_digits=12, decimal_places=4, null=True)
    ema200 = models.DecimalField(max_digits=12, decimal_places=4, null=True)

    class Meta:
        unique_together = ("trade_date", "symbol")
        verbose_name = "Parameter"
        verbose_name_plural = "Parameters"
        ordering = ['trade_date']
        indexes = [
            models.Index(fields=['symbol', 'trade_date']),
            models.Index(fields=['trade_date']),
        ]

    def __str__(self):
        return f"{self.symbol} - {self.trade_date}"



class Index(models.Model):
    name = models.CharField(max_length=100, unique=True)   # Nifty 50, Sensex, Nifty 500
    symbol = models.CharField(max_length=50, unique=True)  # ^NSEI, ^BSESN, NIFTY500
    exchange = models.CharField(max_length=20, null=True, blank=True) # NSE, BSE

    class Meta:
        verbose_name = "Index"
        verbose_name_plural = "Indices"
        ordering = ['name']

    def __str__(self):
        return self.name

class IndexPrice(models.Model):
    index = models.ForeignKey(Index, on_delete=models.CASCADE)
    trade_date = models.DateField()
    open = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    high = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    low = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    close = models.DecimalField(max_digits=12, decimal_places=2)
    volume = models.BigIntegerField(null=True, blank=True)  # Some indices don't have volumes

    class Meta:
        unique_together = ("trade_date", "index")
        verbose_name = "Index Price"
        verbose_name_plural = "Index Prices"
        ordering = ['trade_date']
        indexes = [
            models.Index(fields=['index', 'trade_date']),
            models.Index(fields=['trade_date']),
        ]

    def __str__(self):
        return f"{self.index.name} - {self.trade_date}"

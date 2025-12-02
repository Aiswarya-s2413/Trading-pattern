from django.db import models

class PriceFeature(models.Model):
    """
    Represents one day of calculated features for a stock.
    (This is the main table storing your 10+ years of data.)
    """
    symbol = models.CharField(max_length=50, db_index=True)
    date = models.DateField(db_index=True)
    
    # Core OHLC Data (we'll ignore volume in the model for now)
    open = models.DecimalField(max_digits=10, decimal_places=2)
    high = models.DecimalField(max_digits=10, decimal_places=2)
    low = models.DecimalField(max_digits=10, decimal_places=2)
    close = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Pre-calculated Technical Features
    EMA_50 = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    EMA_200 = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    RSC_20 = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    
    # Pre-calculated Success Metrics (The Filter)
    is_successful_trade = models.DecimalField(max_digits=5, decimal_places=2, default=0.0)

    class Meta:
        unique_together = ('symbol', 'date')
        ordering = ['date']
        
    def __str__(self):
        return f"{self.symbol} - {self.date}"


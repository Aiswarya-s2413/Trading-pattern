# myapp/utils.py
import numpy as np
import pandas as pd
from marketdata.models import EodPrice, Symbol

def get_volatility_metrics(symbol_ticker, days=30):
    """
    Calculates technical volatility metrics.
    Fixes:
    1. 'TCS' vs 'TCS.NS' mismatch.
    2. Decimal vs Float math error.
    """
    try:
        # 1. ROBUST SYMBOL LOOKUP
        symbol_obj = Symbol.objects.filter(symbol=symbol_ticker).first()
        if not symbol_obj:
            symbol_obj = Symbol.objects.filter(symbol=f"{symbol_ticker}.NS").first()
            
        if not symbol_obj:
            print(f"❌ Error: Symbol '{symbol_ticker}' not found.")
            return None
        
        # 2. Fetch Data
        qs = EodPrice.objects.filter(symbol=symbol_obj).order_by('-trade_date')[:days+20]
        data_list = list(qs.values('trade_date', 'high', 'low', 'close'))
        
        if not data_list:
            print(f"❌ Error: No price data for {symbol_ticker}")
            return None

        df = pd.DataFrame(data_list)
        
        # 🔴 CRITICAL FIX: Convert Decimal to Float
        # Django returns Decimals, but Pandas needs Floats for math.
        df[['high', 'low', 'close']] = df[['high', 'low', 'close']].astype(float)

        if len(df) < 15:
            return None

        # Reverse to chronological order (Oldest -> Newest)
        df = df.iloc[::-1].reset_index(drop=True)

        # 3. Calculate Volatility (Std Dev)
        df['returns'] = df['close'].pct_change()
        volatility = df['returns'].std() * np.sqrt(252) * 100 

        # 4. Calculate ATR
        df['prev_close'] = df['close'].shift(1)
        df['h_l'] = df['high'] - df['low']
        df['h_pc'] = abs(df['high'] - df['prev_close'])
        df['l_pc'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        
        current_atr = df['tr'].tail(14).mean()
        current_price = df['close'].iloc[-1]
        
        atr_percentage = (current_atr / current_price) * 100

        return {
            "current_price": round(current_price, 2),
            "annual_volatility_percent": round(volatility, 2),
            "atr_percent": round(atr_percentage, 2),
            "trend_summary": df['close'].tail(5).tolist() 
        }
    except Exception as e:
        print(f"Error calculating volatility: {e}")
        return None
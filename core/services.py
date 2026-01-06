# core/services.py
import os
import json
from openai import OpenAI
from .utils import get_volatility_metrics

def recommend_dip_strategy(symbol_ticker):
    """
    Ultra-Precise Version: Uses a Volatility vs. Range Matrix.
    """
    # 1. Setup
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else None
    
    # 2. Get Data (Auto-Correct .NS)
    metrics = get_volatility_metrics(symbol_ticker)
    if not metrics and not symbol_ticker.endswith(".NS"):
        metrics = get_volatility_metrics(f"{symbol_ticker}.NS")
        
    if not metrics:
        return {"error": "Not enough data"}

    # 3. Extract Precise Variables
    vol = metrics.get('annual_volatility_percent', 20)      # e.g., 25%
    pos = metrics.get('range_position', 50)                 # e.g., 85% (Near High)
    trend = metrics.get('trend_summary', [])                # Last 5 closing prices
    
    # Calculate Momentum (Is it crashing right now?)
    # If Price today < Price 5 days ago, it's weak.
    momentum_score = "Neutral"
    if len(trend) >= 5:
        change_5d = ((trend[-1] - trend[0]) / trend[0]) * 100
        if change_5d > 2: momentum_score = "Strong Uptrend"
        elif change_5d < -2: momentum_score = "Falling"

    # 4. THE MATRIX PROMPT
    prompt = f"""
    You are a Quantitative Algo configuring a 'Volatility Filter'.
    
    INPUT DATA:
    - Stock: {symbol_ticker}
    - Annual Volatility: {vol}%
    - 52-Week Range Position: {pos}% (0=Low, 100=High)
    - Momentum (5-Day): {momentum_score}

    THE DECISION MATRIX (Select the exact cell):

    | Volatility | Range Pos > 80% (Highs) | Range Pos 20-80% (Mid) | Range Pos < 20% (Lows) |
    | :--- | :--- | :--- | :--- |
    | **LOW (<20%)** | Tight (8-12%) | Standard (12-15%) | Value (15-18%) |
    | **MED (20-40%)** | Moderate (12-18%) | Wide (18-25%) | Deep (20-28%) |
    | **HIGH (>40%)** | Loose (18-25%) | Very Wide (25-35%) | Extreme (30-40%) |

    FINE TUNING RULES:
    1. If Momentum is 'Falling', ADD +2% to the matrix value (Be safe).
    2. If Momentum is 'Strong Uptrend', SUBTRACT -2% from the matrix value (Don't miss the bus).
    3. Nifty 50 stocks (Low Vol) should rarely exceed 15%.
    4. Small Caps (High Vol) should rarely go below 20%.

    TASK:
    Output the single, most precise Integer percentage.

    OUTPUT JSON:
    {{
        "recommended_dip_percentage": <integer>,
        "risk_level": "Low/Medium/High",
        "reasoning": "Selected from Matrix [Vol: {vol}, Pos: {pos}]. Adjusted for {momentum_score}."
    }}
    """

    try:
        # 5. EXECUTE AI
        if client:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1 # Very strict, almost robotic
            )
            return json.loads(response.choices[0].message.content)

    except Exception as e:
        print(f"AI Error: {e}")

    # 6. MATHEMATICAL FALLBACK (The "Manual Matrix")
    # If AI fails, we calculate the matrix manually in Python.
    
    # Base = Volatility / 1.6 (Rough approximation of the matrix)
    base = vol / 1.6
    
    # Adjust for Range Position
    if pos > 80: base -= 3  # Near highs? Tighten filter.
    elif pos < 20: base += 2  # Near lows? Widen filter.
    
    # Clamp results
    if vol < 20: base = max(8, min(base, 15))      # Stable stocks: 8-15%
    elif vol > 40: base = max(20, min(base, 35))   # Wild stocks: 20-35%
    else: base = max(12, min(base, 25))            # Normal stocks: 12-25%

    return {
        "recommended_dip_percentage": int(base),
        "risk_level": "Calculated (Fallback)",
        "reasoning": f"Matrix fallback calculation based on {vol}% volatility."
    }
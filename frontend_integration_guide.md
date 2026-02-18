# AI Trading Tool - Frontend Integration Guide

## Overview
This document serves as the blueprint for building the Frontend interface for the AI Trading Tool. It details the Backend API endpoints, Data Models, and the logic required to display predictions and accuracy metrics correctly.

---

## 1. Core Concept
The frontend will display a dashboard where users can see AI-generated stock predictions for the year 2025. The key features are:
1.  **Prediction Table**: A list of stocks with their "Buy/Avoid" signals.
2.  **Confidence Scores**: Visual indicators of how much the AI trusts a trade.
3.  **Accuracy Metrics**: Historical performance of the model for each specific stock.

---

## 2. Backend Data Structure (API Response)

The backend (Django/Python) will provide data in the following JSON structure. You should design your React/Frontend components to consume this format.

### **Endpoint:** `/api/ai-predictions/` (example)
**Method:** `GET`
**Response Format:**

```json
[
  {
    "symbol": "RELIANCE",
    "date": "2025-01-15",
    "sector": "Energy",
    "predicted_label": 1,             // 1 = Buy, 0 = Avoid
    "predicted_probability": 0.42,    // Raw AI Confidence (0.00 - 1.00)
    "sector_confidence": 0.65,        // Sector Win Rate (0.00 - 1.00)
    "sample_confidence": 1.0,         // Statistical Reliability (0.00 - 1.00)
    "stock_accuracy": 0.72            // Model Accuracy for THIS stock (0.00 - 1.00)
  },
  {
    "symbol": "TCS",
    "date": "2025-01-16",
    // ...
  }
]
```

---

## 3. Field Definitions & Display Logic

Here is exactly how to display each field in the Frontend.

### **A. Primary Fields (The Decision)**

| JSON Field | Display Name | UI Logic / Styling |
| :--- | :--- | :--- |
| `symbol` | **Stock** | Bold Text (e.g., **RELIANCE**) |
| `predicted_label` | **Signal** | • If `1`: Show <span style="color:green">**BUY**</span> badge.<br>• If `0`: Show <span style="color:gray">AVOID</span> badge. |
| `predicted_probability` | **AI Score** | Display as a Progress Bar or Percentage.<br>• Range: 0% - 100%.<br>• *Note:* Values > 41% are generally high for this model. |

### **B. Confidence Metrics (The Triple Confluence)**

These 3 metrics help the user decide *how much to trust* the signal. You should interpret them as follows:

| JSON Field | Display Name | Tooltip / Description | Recommended UI |
| :--- | :--- | :--- | :--- |
| `sector_confidence` | **Sector Heat** | "Is the Sector winning right now?" | <span style="color:orange">🔥 Hot</span> if > 0.60 |
| `sample_confidence` | **Data Quality** | "Do we have enough data (30+ trades)?" | <span style="color:blue">ℹ️ High</span> if > 0.80 |
| `stock_accuracy` | **Win Rate** | "How often is the AI right about THIS stock?" | <span style="color:green">✅ Reliable</span> if > 0.70 |

---

## 4. Derived Logic for Frontend

To highlight the **"Best Trades"**, the frontend should implement a filter that highlights rows meeting all three conditions:
1.  `predicted_label` == **1**
2.  `sector_confidence` > **0.60**
3.  `stock_accuracy` > **0.70**

*Suggestion:* Add a toggle button called **"Show High Conviction Only"** that applies this filter.

---

## 5. Technology Stack Reference
*   **Backend:** Python / Django (Views consume `backtest_results_2025_FULL.xlsx`)
*   **AI Model:** PyTorch (LSTM Network)
*   **Data Source:** Historical OHLCV + RSC (Relative Strength)

This guide provides all the necessary context for the frontend team to build the "AI Signal Dashboard" without needing access to the backend code.


import pandas as pd
import numpy as np

file_path = 'ai_training/backtest_results_2025_FULL.xlsx'

try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("Total rows:", len(df))
    
    print("\n--- Statistics ---")
    print(df.describe())
    
    print("\n--- Distribution of Predicted Labels ---")
    print(df['Predicted_Label'].value_counts(normalize=True))
    
    print("\n--- Distribution of Actual Success ---")
    print(df['Actual_Success'].value_counts(normalize=True))
    
    print("\n--- Correlation ---")
    print(df[['Actual_Success', 'Predicted_Probability', 'Sector_Confidence']].corr())
    
    # Check probability stats
    print("\n--- Probability Stats ---")
    print(df['Predicted_Probability'].describe())

    # Check for constant predictions
    unique_probs = df['Predicted_Probability'].nunique()
    print(f"\nUnique Predicted Probabilities: {unique_probs}")
    
    if unique_probs < 10:
        print("WARNING: Very few unique probabilities. Model might be predicting a constant value or failing to learn.")
        print(df['Predicted_Probability'].unique())

except Exception as e:
    print(f"Error reading file: {e}")

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Excel file
file_path = 'NRB_Without_MicroCap.xlsx'

try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# 2. Define Columns
return_col = '12-month %'

if return_col not in df.columns:
    print(f"Error: Column '{return_col}' not found.")
    exit()

# 3. Filter for Success (>= 20%)
# We only want to plot the distribution of the "Successes"
success_df = df[df[return_col] >= 20]

# 4. Plot the Histogram
plt.figure(figsize=(12, 6))

# bins=50 creates 50 bars. You can change this number to make bars wider/thinner.
# range=(20, 300) limits the X-axis to show returns between 20% and 300% (ignoring extreme outliers for clarity)
plt.hist(success_df[return_col], bins=50, range=(20, 300), color='#1f77b4', edgecolor='black', alpha=0.7)

plt.title('Distribution of 12-Month Returns (Success Cases Only)')
plt.xlabel('12-Month Return %')
plt.ylabel('Number of Cases')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
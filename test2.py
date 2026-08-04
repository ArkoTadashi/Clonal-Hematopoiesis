

import pandas as pd

df = pd.read_csv("final_combined_data.csv");
# Returns True if all values match, False if any differ
are_equal = (df["STAG2"] == df["Altered"]).all()
print("Columns are identical:", are_equal)
print(df.notna().sum())
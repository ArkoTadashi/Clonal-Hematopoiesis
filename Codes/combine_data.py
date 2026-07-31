import pandas as pd
from find_files import files


df_combined = None

file_paths = files()
print(file_paths)

for file_path in file_paths:
    
    df_temp = pd.read_csv("../Data/"+file_path+".csv", sep=",")
    df_temp.columns = df_temp.columns.str.strip()

    if df_combined is None:
        df_combined = df_temp
    else:
        df_combined = pd.merge(df_combined, df_temp, on="SAMPLE_ID", how="outer")

df_combined.to_csv("../final_combined_data.csv", index=False)
print("Files successfully merged!")



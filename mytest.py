# import glob
# import pandas as pd

# # 1. Find all .txt files in the folder (or specify a list of filenames)
# file_paths = glob.glob("Data/*.txt")

# df_combined = None

# for file_path in file_paths:
#     # Read the file (change sep='\t' if your files are tab-delimited)
#     df_temp = pd.read_csv(file_path, sep="\t")
#     df_temp.columns = df_temp.columns.str.strip()

#     # If it's the first file, initialize df_combined
#     if df_combined is None:
#         df_combined = df_temp
#     else:
#         # Merge new columns based on SAMPLE_ID
#         df_combined = pd.merge(df_combined, df_temp, on="SAMPLE_ID", how="outer")

# # 2. Export the final combined CSV
# df_combined.to_csv("final_combined_data.csv", index=False)
# print("Files successfully merged!")

import pandas as pd


def convert_tcga_file(input_file_path, output_file_path):
    # Read the file handling flexible space/tab separators
    df = pd.read_csv(input_file_path, sep=r"\s+")

    # Clean leading/trailing whitespaces from column names
    df.columns = df.columns.str.strip()

    # Extract clean SAMPLE_ID by splitting on the colon ':'
    first_col = df.columns[0]  # 'studyID:sampleId'
    df["SAMPLE_ID"] = df[first_col].apply(lambda x: str(x).split(":")[-1])

    # Drop the original 'studyID:sampleId' column
    df = df.drop(columns=[first_col])

    # Reorder columns so SAMPLE_ID is first
    cols = ["SAMPLE_ID"] + [c for c in df.columns if c != "SAMPLE_ID"]
    df = df[cols]

    # Export to a new tab-delimited text file
    df.to_csv(output_file_path, sep="\t", index=False)
    print(f"File saved to {output_file_path}")


# --- Example Usage ---
convert_tcga_file("Original Data/sample_matrix.txt", "Data/sample_matrix.txt");
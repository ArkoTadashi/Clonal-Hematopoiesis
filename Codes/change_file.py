import pandas as pd

def convert_tcga_file(input_file_path, output_file_path):
    # Read the file handling flexible space/tab separators
    df = pd.read_csv(input_file_path, sep="\t", engine="python")

    # Clean leading/trailing whitespaces from column names
    df.columns = df.columns.str.strip()

    first_col = df.columns[0]  # 'studyID:sampleId'

    df = df.drop(columns=[first_col])


    # Export to a new tab-delimited text file
    df.to_csv(output_file_path, sep=",", index=False)
    # df.to_csv(output_file_path, sep=",", index=False, na_rep="NA")
    print(f"File saved to {output_file_path}")

def change(file_names):
    for file_name in file_names:
        if file_name == "altered_samples":
            continue
        if file_name == "unaltered_samples":
            continue
        if file_name == "sample_matrix":
            continue
            
        convert_tcga_file("../Original Data/"+file_name+".txt", "../Data/"+file_name+".csv")


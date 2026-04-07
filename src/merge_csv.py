import pandas as pd
import os
import glob

def merge_all_csv(raw_root, output_path):

    all_data = []

    for folder in os.listdir(raw_root):

        folder_path = os.path.join(raw_root, folder)

        if not os.path.isdir(folder_path):
            continue

        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        if len(csv_files) == 0:
            print(f"No CSV found in {folder}")
            continue

        csv_path = csv_files[0]

        print(f"Reading: {csv_path}")

        df = pd.read_csv(csv_path)

        df = df.rename(columns={
            "Image Data ID": "image_id",
            "Subject": "subject_id",
            "Group": "diagnosis",
            "Age": "age",
            "Sex": "sex"
        })

        df["source_folder"] = folder

        df = df[["image_id","subject_id","diagnosis","age","sex","source_folder"]]

        all_data.append(df)

    if len(all_data) == 0:
        raise ValueError("No CSV files were loaded. Check folder structure.")

    master_df = pd.concat(all_data, ignore_index=True)

    master_df = master_df.dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    master_df.to_csv(output_path, index=False)

    print("Master CSV created successfully")
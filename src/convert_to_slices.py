import nibabel as nib
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from src.utils import get_image_id, get_all_nii_files

def convert_mri_to_slices(mri_root, metadata_path, output_folder, final_csv):

    os.makedirs(output_folder, exist_ok=True)

    metadata = pd.read_csv(metadata_path)

    nii_files = get_all_nii_files(mri_root)
    print("Total MRI files found:", len(nii_files))

    records = []
    counter = 0

    for path in tqdm(nii_files, desc="Processing MRI"):

        filename = os.path.basename(path)

        image_id = get_image_id(filename)

        row = metadata[metadata["image_id"] == image_id]

        if len(row) == 0:
            continue

        try:
            img = nib.load(path)
            data = img.get_fdata()
        except Exception as e:
            print(f"Skipping file: {path}")
            print(f"Reason: {e}")
            continue
        data = img.get_fdata()

        for i in range(40,60):

            slice_img = data[:,:,i]

            slice_img = cv2.resize(slice_img,(224,224))

            slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img)-np.min(slice_img))

            image_name = f"img_{counter}.png"

            cv2.imwrite(os.path.join(output_folder,image_name), slice_img*255)

            records.append([
                image_name,
                image_id,
                row["subject_id"].values[0],
                row["diagnosis"].values[0],
                row["age"].values[0],
                row["sex"].values[0],
                row["source_folder"].values[0]
            ])

            counter += 1

    final_df = pd.DataFrame(records, columns=[
        "image_name","image_id","subject_id","diagnosis","age","sex","source_folder"
    ])

    final_df.to_csv(final_csv, index=False)

    print("Dataset creation complete")
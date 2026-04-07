import zipfile
import os
from tqdm import tqdm

def extract_all_zips(raw_root, output_root):

    os.makedirs(output_root, exist_ok=True)

    for folder in os.listdir(raw_root):

        zip_folder = os.path.join(raw_root, folder, "zips")

        if not os.path.exists(zip_folder):
            continue

        for file in tqdm(os.listdir(zip_folder), desc=f"Extracting {folder}"):

            if file.endswith(".zip"):

                with zipfile.ZipFile(os.path.join(zip_folder,file),'r') as zip_ref:

                    extract_path = os.path.join(output_root, folder)

                    os.makedirs(extract_path, exist_ok=True)

                    zip_ref.extractall(extract_path)

    print("ZIP extraction complete")
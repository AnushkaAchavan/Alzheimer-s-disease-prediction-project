import os

def get_image_id(filename):
    return filename.split("_")[-1].replace(".nii","")

def extract_subject_id(filename):
    return "_".join(filename.split("_")[1:4])

def get_all_nii_files(root_dir):
    nii_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nii_files.append(os.path.join(root,file))
    return nii_files
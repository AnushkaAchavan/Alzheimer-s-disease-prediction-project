from src.extract_zips import extract_all_zips
from src.merge_csv import merge_all_csv
from src.convert_to_slices import convert_mri_to_slices

RAW_DATA = "raw_data"
MRI_OUTPUT = "raw_data"
MASTER_CSV = "processed_data/metadata/master.csv"
SLICE_FOLDER = "processed_data/slices"
FINAL_CSV = "processed_data/metadata/final_metadata.csv"




# Step 2: Merge CSVs
merge_all_csv(RAW_DATA, MASTER_CSV)

# Step 3: Convert MRI to slices + create dataset
convert_mri_to_slices(MRI_OUTPUT, MASTER_CSV, SLICE_FOLDER, FINAL_CSV)
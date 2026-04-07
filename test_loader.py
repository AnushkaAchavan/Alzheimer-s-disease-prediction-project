from dataset.loader import load_dataset

X,y=load_dataset(
    "processed_data/metadata/final_metadata.csv",
    "processed_data/slices"
)
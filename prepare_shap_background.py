from dataset.loader import load_dataset
from explainability.shap_explain import SHAPExplainer

_, val_loader, _, _, _ = load_dataset(
    'processed_data/metadata/final_metadata.csv',
    'processed_data/slices',
    sample_size=500,
    balance=True,
    img_size=256,
    batch_size=8,
    num_workers=0
)

SHAPExplainer.collect_background(
    val_loader,
    n=10,   # smaller for low-memory laptops
    save_path='outputs/shap_background.pt'
)

print("✅ SHAP background saved successfully.")
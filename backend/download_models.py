import os
import boto3
import zipfile

def download_and_extract():
    BUCKET = "avp-rsna-models"
    models_dir = "backend/models"
    os.makedirs(models_dir, exist_ok=True)
    
    s3 = boto3.client('s3')
    
    # List of files to download
    files = ["faster_rcnn_epoch_5.pth", "kaggle_checkpoints.zip"]
    
    for filename in files:
        local_path = os.path.join(models_dir, filename)
        if not os.path.exists(local_path):
            print(f"📦 Downloading {filename}...")
            s3.download_file(BUCKET, filename, local_path)
    
    # --- ZIP EXTRACTION LOGIC ---
    zip_path = os.path.join(models_dir, "kaggle_checkpoints.zip")
    if os.path.exists(zip_path):
        print("📂 Extracting model checkpoints...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        os.remove(zip_path) # Important: saves ~500MB of disk space!
        print("✅ Extraction complete and zip removed.")

if __name__ == "__main__":
    download_and_extract()

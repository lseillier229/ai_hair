import gdown
import zipfile
import os

os.makedirs("dataset_raw", exist_ok=True)

url = "https://drive.google.com/uc?id=0B7EVK8r0v71pMGN3RTU0ZmRteGM"
output = "dataset_raw/celeba.zip"

print("Téléchargement CelebA (1.3GB)...")
gdown.download(url, output, quiet=False)

print("Extraction...")
with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall("dataset_raw")

print("OK ✔️")

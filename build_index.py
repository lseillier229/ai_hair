import os
import faiss
import pickle
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from clip import load as load_clip


# ------------ PARAMS ------------
DATASET_DIR = "hairstyle_images/img_align_celeba"
OUTPUT_DIR = "hairstyle_index"
BATCH_SIZE = 64
IMAGE_EXT = [".jpg", ".jpeg", ".png"]
# --------------------------------


def list_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXT:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def load_and_preprocess(images, preprocess):
    batch = []
    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
            batch.append(preprocess(img))
        except:
            batch.append(None)
    return batch


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📂 Listing images...")
    all_paths = list_images(DATASET_DIR)
    print(f"➡ {len(all_paths)} images trouvées.")

    print("🔠 Chargement du modèle CLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip("ViT-B/32", device=device)

    dim = model.transformer.width  # CLIP embedding dim = 512
    index = faiss.IndexFlatIP(dim)  # Index Similarité cosinus

    print("⚙️ Encodage & construction de l'index...")
    embeddings = []

    for i in tqdm(range(0, len(all_paths), BATCH_SIZE), desc="Processing"):
        batch_paths = all_paths[i:i + BATCH_SIZE]
        
        batch_imgs = load_and_preprocess(batch_paths, preprocess)
        valid_imgs = [img for img in batch_imgs if img is not None]

        if len(valid_imgs) == 0:
            continue

        imgs_tensor = torch.stack(valid_imgs).to(device)

        with torch.no_grad():
            emb = model.encode_image(imgs_tensor).float()
            emb /= emb.norm(dim=-1, keepdim=True)

        emb_np = emb.cpu().numpy()
        index.add(emb_np)

    # Sauvegarde FAISS + chemins
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss.index"))

    with open(os.path.join(OUTPUT_DIR, "paths.pkl"), "wb") as f:
        pickle.dump(all_paths, f)

    print("✅ Index FAISS construit avec succès !")
    print(f"📁 Sauvegardé dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

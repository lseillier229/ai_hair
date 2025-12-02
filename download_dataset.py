# download_dataset.py
"""
Script pour télécharger un dataset de coiffures et construire l'index.
"""
import os
import urllib.request
import zipfile
import shutil

DATASETS = {
    "figaro1k": {
        "url": "https://github.com/elementer/Figaro-1k/archive/refs/heads/master.zip",
        "extract_folder": "Figaro-1k-master",
        "images_subfolder": "Original/Training"
    },
    "lfw": {
        "url": "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz",
        "description": "Labeled Faces in the Wild - 13k+ faces"
    },
    "sample": {
        "description": "Créer un dossier sample avec quelques images manuellement"
    }
}


def download_figaro1k(output_dir="hairstyle_images"):
    """Télécharge le dataset Figaro-1k."""
    
    print("=" * 50)
    print("Téléchargement du dataset Figaro-1k")
    print("=" * 50)
    
    zip_path = "figaro1k.zip"
    url = DATASETS["figaro1k"]["url"]
    
    # Télécharger
    if not os.path.exists(zip_path):
        print(f"[1/4] Téléchargement depuis GitHub...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"      Téléchargé: {zip_path}")
        except Exception as e:
            print(f"      ERREUR: {e}")
            print("\n[ALTERNATIVE] Télécharge manuellement:")
            print(f"   1. Va sur: https://github.com/elementer/Figaro-1k")
            print(f"   2. Code > Download ZIP")
            print(f"   3. Extrais les images dans: {output_dir}/")
            return False
    else:
        print(f"[1/4] Fichier ZIP déjà présent")
    
    # Extraire
    print(f"[2/4] Extraction...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(".")
        print(f"      Extrait.")
    except Exception as e:
        print(f"      ERREUR extraction: {e}")
        return False
    
    # Organiser
    print(f"[3/4] Organisation des images...")
    os.makedirs(output_dir, exist_ok=True)
    
    extract_folder = DATASETS["figaro1k"]["extract_folder"]
    images_subfolder = DATASETS["figaro1k"]["images_subfolder"]
    source_dir = os.path.join(extract_folder, images_subfolder)
    
    if os.path.exists(source_dir):
        count = 0
        for f in os.listdir(source_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(os.path.join(source_dir, f), os.path.join(output_dir, f))
                count += 1
        print(f"      {count} images copiées dans {output_dir}/")
    else:
        # Chercher toutes les images dans l'archive
        print(f"      Recherche d'images dans {extract_folder}...")
        count = 0
        for root, dirs, files in os.walk(extract_folder):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy(os.path.join(root, f), os.path.join(output_dir, f))
                    count += 1
        print(f"      {count} images copiées")
    
    # Nettoyer
    print(f"[4/4] Nettoyage...")
    if os.path.exists(extract_folder):
        shutil.rmtree(extract_folder)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print(f"      Nettoyé.")
    
    return True


def build_search_index(images_dir="hairstyle_images", index_dir="hairstyle_index"):
    """Construit l'index FAISS pour la recherche."""
    
    print("\n" + "=" * 50)
    print("Construction de l'index de recherche")
    print("=" * 50)
    
    if not os.path.exists(images_dir):
        print(f"[ERREUR] Dossier {images_dir}/ non trouvé!")
        print(f"         Place des images de coiffures dedans.")
        return False
    
    # Compter les images
    count = sum(1 for f in os.listdir(images_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
    
    if count == 0:
        print(f"[ERREUR] Aucune image dans {images_dir}/")
        return False
    
    print(f"[INFO] {count} images trouvées dans {images_dir}/")
    print(f"[INFO] Cela peut prendre quelques minutes...\n")
    
    from hairstyle_search import build_index
    build_index(images_dir, index_dir)
    
    print("\n[OK] Index créé avec succès!")
    print(f"     Tu peux maintenant utiliser la recherche dans l'app.")
    return True


def download_lfw(output_dir="hairstyle_images", max_images=500):
    """Télécharge un sous-ensemble du dataset LFW (visages avec coiffures variées)."""
    import tarfile
    
    print("=" * 50)
    print("Téléchargement du dataset LFW (Labeled Faces in the Wild)")
    print("=" * 50)
    
    tgz_path = "lfw-funneled.tgz"
    url = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
    
    # Télécharger
    if not os.path.exists(tgz_path):
        print(f"[1/4] Téléchargement (~233MB, peut prendre quelques minutes)...")
        try:
            urllib.request.urlretrieve(url, tgz_path)
            print(f"      Téléchargé: {tgz_path}")
        except Exception as e:
            print(f"      ERREUR: {e}")
            return False
    else:
        print(f"[1/4] Fichier déjà présent")
    
    # Extraire
    print(f"[2/4] Extraction...")
    try:
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(".")
        print(f"      Extrait.")
    except Exception as e:
        print(f"      ERREUR extraction: {e}")
        return False
    
    # Organiser - prendre un sous-ensemble
    print(f"[3/4] Sélection de {max_images} images...")
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for root, dirs, files in os.walk("lfw_funneled"):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(root, f)
                dst = os.path.join(output_dir, f"{count:04d}_{f}")
                shutil.copy(src, dst)
                count += 1
                if count >= max_images:
                    break
        if count >= max_images:
            break
    
    print(f"      {count} images copiées dans {output_dir}/")
    
    # Nettoyer
    print(f"[4/4] Nettoyage...")
    if os.path.exists("lfw_funneled"):
        shutil.rmtree("lfw_funneled")
    if os.path.exists(tgz_path):
        os.remove(tgz_path)
    print(f"      Nettoyé.")
    
    return True


def download_sample_from_urls(output_dir="hairstyle_images"):
    """Télécharge quelques images d'exemple depuis des URLs publiques."""
    
    print("=" * 50)
    print("Téléchargement d'images d'exemple")
    print("=" * 50)
    
    # URLs d'images libres de droits (Unsplash - 50 images variées)
    sample_urls = [
        # Hommes - cheveux courts
        ("m_short_1.jpg", "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400"),
        ("m_short_2.jpg", "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=400"),
        ("m_short_3.jpg", "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400"),
        ("m_short_4.jpg", "https://images.unsplash.com/photo-1552058544-f2b08422138a?w=400"),
        ("m_short_5.jpg", "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=400"),
        ("m_short_6.jpg", "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400"),
        ("m_short_7.jpg", "https://images.unsplash.com/photo-1568602471122-7832951cc4c5?w=400"),
        ("m_short_8.jpg", "https://images.unsplash.com/photo-1560250097-0b93528c311a?w=400"),
        ("m_short_9.jpg", "https://images.unsplash.com/photo-1566492031773-4f4e44671857?w=400"),
        ("m_short_10.jpg", "https://images.unsplash.com/photo-1583195764036-6dc248ac07d9?w=400"),
        # Hommes - cheveux moyens
        ("m_medium_1.jpg", "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400"),
        ("m_medium_2.jpg", "https://images.unsplash.com/photo-1504257432389-52343af06ae3?w=400"),
        ("m_medium_3.jpg", "https://images.unsplash.com/photo-1480429370612-2f63b11a3e23?w=400"),
        ("m_medium_4.jpg", "https://images.unsplash.com/photo-1519345182560-3f2917c472ef?w=400"),
        ("m_medium_5.jpg", "https://images.unsplash.com/photo-1507591064344-4c6ce005b128?w=400"),
        ("m_medium_6.jpg", "https://images.unsplash.com/photo-1463453091185-61582044d556?w=400"),
        ("m_medium_7.jpg", "https://images.unsplash.com/photo-1548372290-8d01b6c8e78c?w=400"),
        ("m_medium_8.jpg", "https://images.unsplash.com/photo-1531891437562-4301cf35b7e4?w=400"),
        # Hommes - cheveux longs
        ("m_long_1.jpg", "https://images.unsplash.com/photo-1521119989659-a83eee488004?w=400"),
        ("m_long_2.jpg", "https://images.unsplash.com/photo-1595152772835-219674b2a8a6?w=400"),
        ("m_long_3.jpg", "https://images.unsplash.com/photo-1618886614638-80e3c103d31a?w=400"),
        # Hommes - bouclés/frisés
        ("m_curly_1.jpg", "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=400"),
        ("m_curly_2.jpg", "https://images.unsplash.com/photo-1534030347209-467a5b0ad3e6?w=400"),
        ("m_curly_3.jpg", "https://images.unsplash.com/photo-1506277886164-e25aa3f4ef7f?w=400"),
        ("m_curly_4.jpg", "https://images.unsplash.com/photo-1522556189639-b150ed9c4330?w=400"),
        ("m_curly_5.jpg", "https://images.unsplash.com/photo-1496345875659-11f7dd282d1d?w=400"),
        # Femmes - cheveux courts
        ("f_short_1.jpg", "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400"),
        ("f_short_2.jpg", "https://images.unsplash.com/photo-1487412720507-e7ab37603c6f?w=400"),
        ("f_short_3.jpg", "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=400"),
        ("f_short_4.jpg", "https://images.unsplash.com/photo-1580489944761-15a19d654956?w=400"),
        # Femmes - cheveux moyens
        ("f_medium_1.jpg", "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400"),
        ("f_medium_2.jpg", "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=400"),
        ("f_medium_3.jpg", "https://images.unsplash.com/photo-1488426862026-3ee34a7d66df?w=400"),
        ("f_medium_4.jpg", "https://images.unsplash.com/photo-1531123897727-8f129e1688ce?w=400"),
        ("f_medium_5.jpg", "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400"),
        # Femmes - cheveux longs
        ("f_long_1.jpg", "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400"),
        ("f_long_2.jpg", "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400"),
        ("f_long_3.jpg", "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=400"),
        ("f_long_4.jpg", "https://images.unsplash.com/photo-1502823403499-6ccfcf4fb453?w=400"),
        ("f_long_5.jpg", "https://images.unsplash.com/photo-1526510747491-58f928ec870f?w=400"),
        # Femmes - bouclés
        ("f_curly_1.jpg", "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=400"),
        ("f_curly_2.jpg", "https://images.unsplash.com/photo-1509967419530-da38b4704bc6?w=400"),
        ("f_curly_3.jpg", "https://images.unsplash.com/photo-1499557354967-2b2d8910bcca?w=400"),
        ("f_curly_4.jpg", "https://images.unsplash.com/photo-1523264653568-d3d4032d1476?w=400"),
        # Divers styles
        ("style_1.jpg", "https://images.unsplash.com/photo-1507081323647-4d250478b919?w=400"),
        ("style_2.jpg", "https://images.unsplash.com/photo-1513956589380-bad6acb9b9d4?w=400"),
        ("style_3.jpg", "https://images.unsplash.com/photo-1519058082700-08a0b56da9b4?w=400"),
        ("style_4.jpg", "https://images.unsplash.com/photo-1504593811423-6dd665756598?w=400"),
        ("style_5.jpg", "https://images.unsplash.com/photo-1528892952291-009c663ce843?w=400"),
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for filename, url in sample_urls:
        try:
            filepath = os.path.join(output_dir, filename)
            print(f"  Téléchargement: {filename}...")
            urllib.request.urlretrieve(url, filepath)
            count += 1
        except Exception as e:
            print(f"  Erreur {filename}: {e}")
    
    print(f"\n✅ {count} images téléchargées dans {output_dir}/")
    return count > 0


if __name__ == "__main__":
    import sys
    
    print("\n🎨 Setup du dataset de coiffures\n")
    
    # Option 1: Télécharger Figaro-1k
    if len(sys.argv) > 1 and sys.argv[1] == "--figaro":
        success = download_figaro1k()
        if success:
            build_search_index()
    
    # Option 2: Télécharger LFW (alternative)
    elif len(sys.argv) > 1 and sys.argv[1] == "--lfw":
        success = download_lfw(max_images=500)
        if success:
            build_search_index()
    
    # Option 3: Télécharger quelques samples (rapide)
    elif len(sys.argv) > 1 and sys.argv[1] == "--sample":
        success = download_sample_from_urls()
        if success:
            build_search_index()
    
    # Option 4: Juste construire l'index (images déjà présentes)
    elif len(sys.argv) > 1 and sys.argv[1] == "--index":
        build_search_index()
    
    # Option 5: Instructions
    else:
        print("Usage:")
        print("  python download_dataset.py --sample   # Télécharger 10 images d'exemple (rapide)")
        print("  python download_dataset.py --lfw      # Télécharger 500 images LFW (~233MB)")
        print("  python download_dataset.py --figaro   # Télécharger Figaro-1k (peut échouer)")
        print("  python download_dataset.py --index    # Indexer des images existantes")
        print("")
        print("Recommandé: commence par --sample pour tester rapidement")

# hairstyle_search.py
"""
Module de recherche de coiffures par similarité avec CLIP + FAISS
"""
import os
import pickle
import numpy as np
from PIL import Image

# Lazy imports pour éviter les erreurs si pas installé
_clip_model = None
_clip_processor = None
_faiss_index = None
_image_paths = []
_embeddings_loaded = False


def _load_clip():
    """Charge le modèle CLIP (une seule fois)."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        print("[CLIP] Chargement du modèle...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("[CLIP] Modèle chargé.")
    return _clip_model, _clip_processor


def get_image_embedding(image):
    """
    Génère l'embedding CLIP d'une image.
    
    Args:
        image: PIL.Image ou np.ndarray (RGB)
    
    Returns:
        np.ndarray de shape (512,)
    """
    import torch
    model, processor = _load_clip()
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    
    # Normaliser l'embedding
    embedding = features.numpy().flatten()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def build_index(image_folder, output_path="hairstyle_index"):
    """
    Construit l'index FAISS à partir d'un dossier d'images.
    
    Args:
        image_folder: Chemin vers le dossier contenant les images
        output_path: Chemin de sortie pour l'index
    """
    import faiss
    
    print(f"[INDEX] Scan du dossier: {image_folder}")
    
    # Collecter toutes les images
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_paths = []
    
    for root, dirs, files in os.walk(image_folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(root, f))
    
    print(f"[INDEX] {len(image_paths)} images trouvées")
    
    if len(image_paths) == 0:
        print("[INDEX] Aucune image trouvée!")
        return
    
    # Générer les embeddings
    embeddings = []
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            emb = get_image_embedding(img)
            embeddings.append(emb)
            if (i + 1) % 10 == 0:
                print(f"[INDEX] {i + 1}/{len(image_paths)} images traitées")
        except Exception as e:
            print(f"[INDEX] Erreur sur {path}: {e}")
            image_paths[i] = None
    
    # Filtrer les erreurs
    valid_pairs = [(p, e) for p, e in zip(image_paths, embeddings) if p is not None]
    image_paths = [p for p, e in valid_pairs]
    embeddings = [e for p, e in valid_pairs]
    
    # Créer l'index FAISS
    embeddings_matrix = np.array(embeddings).astype('float32')
    dimension = embeddings_matrix.shape[1]
    
    index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity car normalisé)
    index.add(embeddings_matrix)
    
    print(f"[INDEX] Index créé avec {index.ntotal} vecteurs")
    
    # Sauvegarder
    os.makedirs(output_path, exist_ok=True)
    faiss.write_index(index, os.path.join(output_path, "faiss.index"))
    with open(os.path.join(output_path, "paths.pkl"), "wb") as f:
        pickle.dump(image_paths, f)
    
    print(f"[INDEX] Sauvegardé dans {output_path}/")


def load_index(index_path="hairstyle_index"):
    """Charge l'index FAISS depuis le disque."""
    global _faiss_index, _image_paths, _embeddings_loaded
    
    if _embeddings_loaded:
        return True
    
    import faiss
    
    index_file = os.path.join(index_path, "faiss.index")
    paths_file = os.path.join(index_path, "paths.pkl")
    
    if not os.path.exists(index_file) or not os.path.exists(paths_file):
        print(f"[SEARCH] Index non trouvé dans {index_path}/")
        return False
    
    _faiss_index = faiss.read_index(index_file)
    with open(paths_file, "rb") as f:
        _image_paths = pickle.load(f)
    
    # Normaliser les chemins (Windows/Linux compatibilité)
    _image_paths = [p.replace("\\", "/") for p in _image_paths]
    
    _embeddings_loaded = True
    print(f"[SEARCH] Index chargé: {_faiss_index.ntotal} coiffures")
    return True


def search_similar(image, top_k=5, index_path="hairstyle_index"):
    """
    Recherche les coiffures les plus similaires à une image.
    
    Args:
        image: PIL.Image ou np.ndarray (RGB) - idéalement la zone cheveux
        top_k: Nombre de résultats à retourner
        index_path: Chemin vers l'index FAISS
    
    Returns:
        Liste de tuples (chemin_image, score_similarité)
    """
    global _faiss_index, _image_paths
    
    if not load_index(index_path):
        return []
    
    # Générer l'embedding de la requête
    query_embedding = get_image_embedding(image).reshape(1, -1).astype('float32')
    
    # Rechercher
    scores, indices = _faiss_index.search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(_image_paths):
            results.append((_image_paths[idx], float(score)))
    
    return results


def search_by_text(text_query, top_k=5, index_path="hairstyle_index"):
    """
    Recherche des coiffures par description textuelle (CLIP text encoder).
    
    Args:
        text_query: Description textuelle (ex: "short curly hair")
        top_k: Nombre de résultats
    
    Returns:
        Liste de tuples (chemin_image, score)
    """
    import torch
    global _faiss_index, _image_paths
    
    if not load_index(index_path):
        return []
    
    model, processor = _load_clip()
    
    # Encoder le texte
    inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    query_embedding = text_features.numpy().flatten()
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Rechercher
    scores, indices = _faiss_index.search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(_image_paths):
            results.append((_image_paths[idx], float(score)))
    
    return results


# ====== RECOMMANDATIONS INTELLIGENTES ======

# Styles recommandés par forme de visage (pour requêtes CLIP)
FACE_SHAPE_STYLE_QUERIES = {
    "oval": [
        "pompadour hairstyle man",
        "quiff textured hair",
        "side part classic haircut",
        "medium length wavy hair",
        "slicked back hair"
    ],
    "round": [
        "high pompadour volume top",
        "faux hawk hairstyle",
        "spiky textured hair top",
        "angular fringe haircut",
        "undercut high volume"
    ],
    "square": [
        "textured crop haircut",
        "messy fringe hairstyle",
        "medium length soft hair",
        "layered haircut men",
        "natural wavy hair"
    ],
    "heart": [
        "side swept fringe hair",
        "medium textured haircut",
        "chin length hair men",
        "layered side part",
        "soft fringe hairstyle"
    ],
    "diamond": [
        "textured fringe haircut",
        "side swept bangs men",
        "medium crop hairstyle",
        "soft layers haircut",
        "curtain bangs men"
    ],
    "oblong": [
        "thick fringe haircut",
        "side volume hairstyle",
        "wavy textured hair",
        "curly medium hair",
        "layered bangs haircut"
    ]
}


def recommend_for_face_shape(face_shape, current_image=None, top_k=6, index_path="hairstyle_index"):
    """
    Recommande des coiffures adaptées à une forme de visage,
    en excluant les styles trop similaires à l'image actuelle.
    
    Args:
        face_shape: Forme du visage détectée (oval, round, square, etc.)
        current_image: Image actuelle (pour exclure les similaires)
        top_k: Nombre de recommandations
        index_path: Chemin vers l'index
    
    Returns:
        Liste de tuples (chemin_image, score, style_name)
    """
    import torch
    global _faiss_index, _image_paths
    
    if not load_index(index_path):
        return []
    
    # Obtenir les styles recommandés pour cette forme
    style_queries = FACE_SHAPE_STYLE_QUERIES.get(face_shape, FACE_SHAPE_STYLE_QUERIES["oval"])
    
    # Embedding de l'image actuelle (pour filtrer les similaires)
    current_embedding = None
    if current_image is not None:
        current_embedding = get_image_embedding(current_image)
    
    model, processor = _load_clip()
    
    all_results = []
    seen_paths = set()
    
    for style_query in style_queries:
        # Encoder la requête textuelle du style
        inputs = processor(text=[style_query], return_tensors="pt", padding=True)
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        
        query_embedding = text_features.numpy().flatten()
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Rechercher plus de résultats pour pouvoir filtrer
        scores, indices = _faiss_index.search(query_embedding, top_k * 3)
        
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(_image_paths):
                continue
                
            img_path = _image_paths[idx]
            
            # Éviter les doublons
            if img_path in seen_paths:
                continue
            
            # Filtrer les images trop similaires à l'actuelle
            if current_embedding is not None:
                try:
                    candidate_img = Image.open(img_path).convert("RGB")
                    candidate_embedding = get_image_embedding(candidate_img)
                    similarity = np.dot(current_embedding, candidate_embedding)
                    
                    # Si trop similaire (>0.85), on skip
                    if similarity > 0.85:
                        continue
                except:
                    pass
            
            seen_paths.add(img_path)
            # Extraire le nom du style de la requête
            style_name = style_query.replace(" men", "").replace(" man", "").title()
            all_results.append((img_path, float(score), style_name))
    
    # Trier par score et retourner les top_k
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[:top_k]


def recommend_different_styles(current_image, face_shape="oval", top_k=6, index_path="hairstyle_index", 
                               hair_analysis=None, custom_queries=None):
    """
    Recommande des coiffures adaptées à la forme du visage et aux caractéristiques des cheveux.
    
    Args:
        current_image: Image actuelle (PIL ou np.ndarray)
        face_shape: Forme du visage
        top_k: Nombre de résultats
        index_path: Chemin vers l'index FAISS
        hair_analysis: Dict avec longueur, texture, couleur (optionnel)
        custom_queries: Liste de requêtes personnalisées (optionnel)
    
    Returns:
        Liste de tuples (chemin_image, score, style_suggéré)
    """
    global _faiss_index, _image_paths
    
    if not load_index(index_path):
        return []
    
    # Utiliser les requêtes personnalisées si fournies, sinon les requêtes par défaut
    if custom_queries:
        style_queries = custom_queries
    else:
        style_queries = FACE_SHAPE_STYLE_QUERIES.get(face_shape, FACE_SHAPE_STYLE_QUERIES["oval"])
    
    import torch
    model, processor = _load_clip()
    
    all_results = []
    seen_paths = set()
    
    # Pour chaque style recommandé, chercher les images correspondantes
    for style_query in style_queries:
        inputs = processor(text=[style_query], return_tensors="pt", padding=True)
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        
        query_embedding = text_features.numpy().flatten()
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Chercher dans l'index
        k = min(top_k, _faiss_index.ntotal)
        scores, indices = _faiss_index.search(query_embedding, k)
        
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(_image_paths):
                continue
            
            img_path = _image_paths[idx]
            
            if img_path in seen_paths:
                continue
            
            seen_paths.add(img_path)
            style_name = style_query.replace(" men", "").replace(" man", "").title()
            all_results.append((img_path, float(score), style_name))
    
    # Trier par score et retourner les meilleurs
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[:top_k]

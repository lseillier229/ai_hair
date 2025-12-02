# hair_analysis.py
"""
Module d'analyse des caractéristiques des cheveux.
Détecte la longueur, texture et couleur des cheveux.
"""
import cv2
import numpy as np


def analyze_hair(image_bgr, hair_mask, face_landmarks=None):
    """
    Analyse les caractéristiques des cheveux.
    
    Args:
        image_bgr: Image BGR
        hair_mask: Masque des cheveux (0-1)
        face_landmarks: Points du visage (optionnel)
    
    Returns:
        dict avec longueur, texture, couleur
    """
    results = {
        "length": "medium",
        "texture": "straight",
        "color": "brown",
        "color_rgb": (100, 80, 60)
    }
    
    if hair_mask is None or hair_mask.max() < 0.1:
        return results
    
    h, w = image_bgr.shape[:2]
    
    # Extraire la région des cheveux
    mask_binary = (hair_mask > 0.5).astype(np.uint8)
    hair_pixels = image_bgr[mask_binary > 0]
    
    if len(hair_pixels) < 100:
        return results
    
    # 1. Analyser la COULEUR
    results["color"], results["color_rgb"] = _analyze_color(hair_pixels)
    
    # 2. Analyser la LONGUEUR
    results["length"] = _analyze_length(mask_binary, face_landmarks, h, w)
    
    # 3. Analyser la TEXTURE
    results["texture"] = _analyze_texture(image_bgr, mask_binary)
    
    return results


def _analyze_color(hair_pixels):
    """Détermine la couleur dominante des cheveux."""
    # Convertir en LAB pour mieux analyser les couleurs
    hair_pixels_reshaped = hair_pixels.reshape(-1, 1, 3).astype(np.uint8)
    lab_pixels = cv2.cvtColor(hair_pixels_reshaped, cv2.COLOR_BGR2LAB)
    
    # Moyenne des couleurs
    mean_bgr = np.mean(hair_pixels, axis=0)
    mean_lab = np.mean(lab_pixels.reshape(-1, 3), axis=0)
    
    L, a, b = mean_lab  # L=luminosité, a=vert-rouge, b=bleu-jaune
    
    # Classification basée sur la luminosité et les teintes
    if L < 40:
        color = "black"
    elif L < 80:
        if b > 135:  # Teinte jaune/orange
            color = "auburn"
        elif a > 135:  # Teinte rouge
            color = "red"
        else:
            color = "brown"
    elif L < 150:
        if b > 140:
            color = "light_brown"
        else:
            color = "dark_blonde"
    else:
        color = "blonde"
    
    # Détecter le gris/blanc
    saturation = np.std(hair_pixels, axis=1).mean()
    if saturation < 15 and L > 100:
        color = "gray"
    
    return color, tuple(int(c) for c in mean_bgr)


def _analyze_length(mask_binary, face_landmarks, h, w):
    """Détermine la longueur des cheveux."""
    # Trouver les limites des cheveux
    hair_coords = np.where(mask_binary > 0)
    if len(hair_coords[0]) == 0:
        return "medium"
    
    hair_top = hair_coords[0].min()
    hair_bottom = hair_coords[0].max()
    hair_height = hair_bottom - hair_top
    
    # Si on a les landmarks, comparer avec la position du visage
    if face_landmarks and len(face_landmarks) > 152:
        chin_y = face_landmarks[152][1]  # Point du menton
        forehead_y = face_landmarks[10][1]  # Point du front
        face_height = chin_y - forehead_y
        
        # Ratio cheveux / visage
        if hair_bottom > chin_y + face_height * 0.5:
            return "long"
        elif hair_bottom > chin_y:
            return "medium"
        else:
            return "short"
    
    # Sans landmarks, utiliser des heuristiques
    hair_ratio = hair_height / h
    
    if hair_ratio > 0.5:
        return "long"
    elif hair_ratio > 0.3:
        return "medium"
    else:
        return "short"


def _analyze_texture(image_bgr, mask_binary):
    """Détermine la texture des cheveux (lisse, ondulé, bouclé)."""
    # Extraire la région des cheveux
    hair_region = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_binary)
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un filtre de détection de contours
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=mask_binary)
    
    # Calculer la densité de contours (indicateur de texture)
    edge_density = np.sum(edges > 0) / max(np.sum(mask_binary > 0), 1)
    
    # Analyser la variance locale (cheveux bouclés = plus de variance)
    kernel_size = 15
    local_var = cv2.blur(gray.astype(np.float32) ** 2, (kernel_size, kernel_size)) - \
                cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size)) ** 2
    
    mean_variance = np.mean(local_var[mask_binary > 0]) if np.sum(mask_binary > 0) > 0 else 0
    
    # Classification
    if edge_density > 0.15 or mean_variance > 800:
        return "curly"
    elif edge_density > 0.08 or mean_variance > 400:
        return "wavy"
    else:
        return "straight"


def build_clip_query(face_shape, hair_analysis, gender="neutral"):
    """
    Construit une requête CLIP personnalisée.
    
    Args:
        face_shape: Forme du visage (oval, round, square...)
        hair_analysis: Résultat de analyze_hair()
        gender: "male", "female", ou "neutral"
    
    Returns:
        Liste de requêtes texte pour CLIP
    """
    length = hair_analysis.get("length", "medium")
    texture = hair_analysis.get("texture", "straight")
    color = hair_analysis.get("color", "brown")
    
    # Mapping des styles recommandés par forme
    style_keywords = {
        "oval": ["classic", "versatile", "any style"],
        "round": ["volume on top", "height", "angular"],
        "square": ["soft", "textured", "layered"],
        "heart": ["side swept", "chin length", "volume at jaw"],
        "diamond": ["fringe", "side part", "width at forehead"],
        "oblong": ["bangs", "side volume", "width"]
    }
    
    # Mapping texture
    texture_words = {
        "straight": "straight sleek",
        "wavy": "wavy textured",
        "curly": "curly natural"
    }
    
    # Mapping longueur
    length_words = {
        "short": "short",
        "medium": "medium length",
        "long": "long"
    }
    
    # Mapping couleur
    color_words = {
        "black": "dark black",
        "brown": "brown",
        "light_brown": "light brown",
        "dark_blonde": "dark blonde", 
        "blonde": "blonde",
        "red": "red ginger",
        "auburn": "auburn",
        "gray": "gray silver"
    }
    
    # Construire les requêtes
    base_style = style_keywords.get(face_shape, style_keywords["oval"])
    tex = texture_words.get(texture, "")
    lng = length_words.get(length, "medium")
    col = color_words.get(color, "")
    
    gender_word = ""
    if gender == "male":
        gender_word = "man male"
    elif gender == "female":
        gender_word = "woman female"
    
    queries = []
    for style in base_style[:3]:  # Top 3 styles
        query = f"{lng} {tex} {col} hair {style} hairstyle {gender_word}".strip()
        query = " ".join(query.split())  # Nettoyer les espaces
        queries.append(query)
    
    # Ajouter des requêtes génériques
    queries.append(f"{lng} {tex} hair hairstyle portrait")
    queries.append(f"stylish {lng} hair {tex}")
    
    return queries


def get_hair_description(hair_analysis):
    """Retourne une description lisible des cheveux."""
    length = hair_analysis.get("length", "medium")
    texture = hair_analysis.get("texture", "straight")
    color = hair_analysis.get("color", "brown")
    
    length_fr = {"short": "courts", "medium": "mi-longs", "long": "longs"}
    texture_fr = {"straight": "lisses", "wavy": "ondulés", "curly": "bouclés"}
    color_fr = {
        "black": "noirs", "brown": "bruns", "light_brown": "châtains",
        "dark_blonde": "blond foncé", "blonde": "blonds",
        "red": "roux", "auburn": "auburn", "gray": "gris"
    }
    
    return f"Cheveux {length_fr.get(length, length)}, {texture_fr.get(texture, texture)}, {color_fr.get(color, color)}"

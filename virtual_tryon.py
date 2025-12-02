# virtual_tryon.py
"""
Module d'essayage virtuel de coiffures.
Superpose des overlays de coiffures sur la photo de l'utilisateur.
"""
import cv2
import numpy as np
from PIL import Image
import os

# Dossier des overlays
OVERLAYS_DIR = "hairstyle_overlays"

# Catalogue des overlays disponibles
OVERLAY_CATALOG = {
    "pompadour": {
        "name": "Pompadour",
        "file": "pompadour.png",
        "offset_y": -1.0,   # Décalage vertical (% de la hauteur de l'overlay, négatif = au-dessus)
        "scale": 1.4,       # Échelle par rapport à la largeur du visage
        "description": "Volume sur le dessus, côtés courts"
    },
    "quiff": {
        "name": "Quiff",
        "file": "quiff.png",
        "offset_y": -0.9,
        "scale": 1.3,
        "description": "Mèche volumineuse vers l'arrière"
    },
    "buzz_cut": {
        "name": "Buzz Cut",
        "file": "buzz_cut.png",
        "offset_y": -0.7,
        "scale": 1.2,
        "description": "Coupe très courte uniforme"
    },
    "side_part": {
        "name": "Side Part",
        "file": "side_part.png",
        "offset_y": -0.85,
        "scale": 1.3,
        "description": "Raie sur le côté, classique"
    },
    "curly": {
        "name": "Cheveux bouclés",
        "file": "curly.png",
        "offset_y": -0.9,
        "scale": 1.5,
        "description": "Boucles naturelles volumineuses"
    },
    "long_straight": {
        "name": "Cheveux longs",
        "file": "long_straight.png",
        "offset_y": -0.6,
        "scale": 1.6,
        "description": "Cheveux longs et lisses"
    }
}


def get_available_overlays():
    """Retourne la liste des overlays disponibles."""
    available = []
    for key, info in OVERLAY_CATALOG.items():
        overlay_path = os.path.join(OVERLAYS_DIR, info["file"])
        exists = os.path.exists(overlay_path)
        available.append({
            "id": key,
            "name": info["name"],
            "description": info["description"],
            "available": exists,
            "path": overlay_path
        })
    return available


def apply_hairstyle_overlay(image, face_landmarks, hairstyle_id, hair_color=None):
    """
    Applique un overlay de coiffure sur l'image.
    
    Args:
        image: Image BGR (numpy array)
        face_landmarks: Points du visage MediaPipe
        hairstyle_id: ID de la coiffure dans OVERLAY_CATALOG
        hair_color: Couleur optionnelle (B, G, R) pour teinter l'overlay
    
    Returns:
        Image avec l'overlay appliqué
    """
    if hairstyle_id not in OVERLAY_CATALOG:
        return image
    
    config = OVERLAY_CATALOG[hairstyle_id]
    overlay_path = os.path.join(OVERLAYS_DIR, config["file"])
    
    if not os.path.exists(overlay_path):
        return image
    
    # Charger l'overlay avec transparence
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return image
    
    # Extraire les dimensions du visage depuis les landmarks
    h, w = image.shape[:2]
    
    # Points clés pour positionner l'overlay
    # 10 = front, 152 = menton, 234 = joue gauche, 454 = joue droite
    forehead = face_landmarks[10]
    chin = face_landmarks[152]
    left_cheek = face_landmarks[234]
    right_cheek = face_landmarks[454]
    
    # Calculer les dimensions
    face_width = abs(right_cheek[0] - left_cheek[0])
    face_height = abs(chin[1] - forehead[1])
    face_center_x = (left_cheek[0] + right_cheek[0]) / 2
    
    # Redimensionner l'overlay
    overlay_width = int(face_width * config["scale"])
    overlay_height = int(overlay_width * overlay.shape[0] / overlay.shape[1])
    overlay_resized = cv2.resize(overlay, (overlay_width, overlay_height))
    
    # Position de l'overlay - AU-DESSUS du front
    x = int(face_center_x - overlay_width / 2)
    # Le front est le point le plus haut, on place l'overlay au-dessus
    y = int(forehead[1] + overlay_height * config["offset_y"])
    
    # Appliquer la couleur si spécifiée
    if hair_color is not None and overlay_resized.shape[2] == 4:
        overlay_resized = tint_overlay(overlay_resized, hair_color)
    
    # Fusionner l'overlay avec l'image
    result = overlay_image(image, overlay_resized, x, y)
    
    return result


def tint_overlay(overlay, color):
    """
    Teinte un overlay PNG avec une couleur.
    
    Args:
        overlay: Image BGRA
        color: Tuple (B, G, R)
    
    Returns:
        Overlay teinté
    """
    # Séparer les canaux
    b, g, r, a = cv2.split(overlay)
    
    # Créer une version teintée
    tinted = overlay.copy()
    
    # Appliquer la teinte (mélange avec la couleur)
    blend_factor = 0.5
    tinted[:, :, 0] = np.clip(b * (1 - blend_factor) + color[0] * blend_factor, 0, 255).astype(np.uint8)
    tinted[:, :, 1] = np.clip(g * (1 - blend_factor) + color[1] * blend_factor, 0, 255).astype(np.uint8)
    tinted[:, :, 2] = np.clip(r * (1 - blend_factor) + color[2] * blend_factor, 0, 255).astype(np.uint8)
    
    return tinted


def overlay_image(background, overlay, x, y):
    """
    Superpose une image avec transparence sur un fond.
    
    Args:
        background: Image de fond BGR
        overlay: Image à superposer BGRA (avec alpha)
        x, y: Position du coin supérieur gauche
    
    Returns:
        Image fusionnée
    """
    result = background.copy()
    h, w = background.shape[:2]
    oh, ow = overlay.shape[:2]
    
    # Calculer les zones de chevauchement
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + ow), min(h, y + oh)
    
    # Zones correspondantes dans l'overlay
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return result
    
    # Extraire la région
    overlay_region = overlay[oy1:oy2, ox1:ox2]
    background_region = result[y1:y2, x1:x2]
    
    if overlay_region.shape[2] == 4:
        # Avec canal alpha
        alpha = overlay_region[:, :, 3:4] / 255.0
        rgb = overlay_region[:, :, :3]
        
        # Blending
        blended = (rgb * alpha + background_region * (1 - alpha)).astype(np.uint8)
        result[y1:y2, x1:x2] = blended
    else:
        # Sans alpha, copie directe
        result[y1:y2, x1:x2] = overlay_region[:, :, :3]
    
    return result


def create_sample_overlay(output_path, style="pompadour", size=(400, 300)):
    """
    Crée un overlay de démonstration (forme simple).
    Pour la production, utiliser de vrais PNG transparents.
    """
    w, h = size
    img = np.zeros((h, w, 4), dtype=np.uint8)
    
    if style == "pompadour":
        # Forme de pompadour simplifiée
        cv2.ellipse(img, (w//2, h//2 + 50), (w//3, h//3), 0, 180, 360, (60, 40, 30, 255), -1)
        cv2.ellipse(img, (w//2, h//2 - 20), (w//4, h//4), 0, 0, 180, (80, 50, 40, 255), -1)
    elif style == "quiff":
        cv2.ellipse(img, (w//2, h//2 + 30), (w//3, h//3), 0, 180, 360, (70, 45, 35, 255), -1)
        pts = np.array([[w//2 - 80, h//2], [w//2, h//4], [w//2 + 80, h//2]], np.int32)
        cv2.fillPoly(img, [pts], (90, 55, 45, 255))
    elif style == "buzz_cut":
        cv2.ellipse(img, (w//2, h//2 + 20), (w//3, h//3 - 30), 0, 180, 360, (50, 35, 25, 255), -1)
    elif style == "side_part":
        cv2.ellipse(img, (w//2 + 20, h//2 + 40), (w//3, h//3), 0, 180, 360, (65, 42, 32, 255), -1)
        cv2.ellipse(img, (w//2 - 30, h//2), (w//5, h//5), -20, 0, 180, (85, 52, 42, 255), -1)
    elif style == "curly":
        # Boucles positionnées en haut de l'image (pour être au-dessus du front)
        for i in range(6):
            cx = w//5 + (i % 3) * w//4
            cy = h//6 + (i // 3) * h//5
            cv2.circle(img, (cx, cy), 40, (75, 48, 38, 255), -1)
    elif style == "long_straight":
        cv2.ellipse(img, (w//2, h//4), (w//3, h//6), 0, 0, 180, (70, 45, 35, 255), -1)
        cv2.rectangle(img, (w//4, h//4), (3*w//4, h), (70, 45, 35, 255), -1)
    
    # Adoucir les bords
    img[:, :, 3] = cv2.GaussianBlur(img[:, :, 3], (15, 15), 0)
    
    cv2.imwrite(output_path, img)
    return output_path


def setup_demo_overlays():
    """Crée des overlays de démonstration si le dossier est vide."""
    os.makedirs(OVERLAYS_DIR, exist_ok=True)
    
    created = []
    for style_id, config in OVERLAY_CATALOG.items():
        path = os.path.join(OVERLAYS_DIR, config["file"])
        if not os.path.exists(path):
            create_sample_overlay(path, style=style_id)
            created.append(style_id)
    
    return created


if __name__ == "__main__":
    print("Création des overlays de démonstration...")
    created = setup_demo_overlays()
    print(f"Overlays créés: {created}")
    print(f"Dossier: {OVERLAYS_DIR}/")

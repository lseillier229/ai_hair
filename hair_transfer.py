# hair_transfer.py
"""
Module de transfert de coiffure local.
Extrait les cheveux d'une image source et les applique sur une image cible.
"""
import cv2
import numpy as np
from PIL import Image
import os

# Réutiliser BiSeNet si disponible
try:
    import onnxruntime as ort
    HAS_ORT = True
except:
    HAS_ORT = False


class HairTransfer:
    """Transfert de coiffure d'une image source vers une image cible."""
    
    def __init__(self, bisenet_path="models/bisenet_faceparsing.onnx"):
        self.bisenet = None
        self.bisenet_path = bisenet_path
        
        if HAS_ORT and os.path.exists(bisenet_path):
            try:
                self._load_bisenet()
            except Exception as e:
                print(f"[HairTransfer] BiSeNet non chargé: {e}")
    
    def _load_bisenet(self):
        """Charge le modèle BiSeNet pour la segmentation."""
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = max(1, os.cpu_count() // 2)
        
        self.bisenet = ort.InferenceSession(
            self.bisenet_path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )
        
        inp = self.bisenet.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = 512
        
        outs = self.bisenet.get_outputs()
        self.output_name = outs[0].name
    
    def _segment_hair(self, image_bgr, hair_class_id=13):
        """Segmente les cheveux d'une image."""
        if self.bisenet is None:
            return None
        
        h, w = image_bgr.shape[:2]
        
        # Prétraitement
        img = cv2.resize(image_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
        
        # Inférence
        output = self.bisenet.run([self.output_name], {self.input_name: img})[0]
        
        # Post-traitement
        if output.ndim == 4:
            pred = np.argmax(output, axis=1)[0]
        else:
            pred = output[0]
        
        # Masque cheveux
        hair_mask = (pred == hair_class_id).astype(np.float32)
        hair_mask = cv2.resize(hair_mask, (w, h))
        hair_mask = cv2.GaussianBlur(hair_mask, (5, 5), 0)
        
        return hair_mask
    
    def _get_face_bounds(self, landmarks):
        """Calcule les limites du visage depuis les landmarks."""
        if landmarks is None or len(landmarks) < 468:
            return None
        
        # Points clés
        forehead = landmarks[10]
        chin = landmarks[152]
        left = landmarks[234]
        right = landmarks[454]
        
        return {
            "center_x": (left[0] + right[0]) // 2,
            "center_y": (forehead[1] + chin[1]) // 2,
            "width": abs(right[0] - left[0]),
            "height": abs(chin[1] - forehead[1]),
            "top": forehead[1]
        }
    
    def transfer(self, target_image, source_image, target_landmarks, source_landmarks=None):
        """
        Transfère les cheveux de source_image vers target_image.
        
        Args:
            target_image: Image cible BGR (ta photo)
            source_image: Image source BGR (coiffure à copier)
            target_landmarks: Landmarks MediaPipe de la cible
            source_landmarks: Landmarks MediaPipe de la source (optionnel)
        
        Returns:
            Image avec les cheveux transférés
        """
        if self.bisenet is None:
            print("[HairTransfer] BiSeNet non disponible")
            return target_image
        
        # Segmenter les cheveux de la source
        source_hair_mask = self._segment_hair(source_image)
        if source_hair_mask is None:
            return target_image
        
        # Segmenter les cheveux de la cible (pour les remplacer)
        target_hair_mask = self._segment_hair(target_image)
        
        # Obtenir les dimensions du visage cible
        target_bounds = self._get_face_bounds(target_landmarks)
        if target_bounds is None:
            return target_image
        
        # Extraire la région des cheveux de la source
        source_h, source_w = source_image.shape[:2]
        target_h, target_w = target_image.shape[:2]
        
        # Trouver la bounding box des cheveux dans la source
        hair_coords = np.where(source_hair_mask > 0.5)
        if len(hair_coords[0]) == 0:
            return target_image
        
        y_min, y_max = hair_coords[0].min(), hair_coords[0].max()
        x_min, x_max = hair_coords[1].min(), hair_coords[1].max()
        
        # Extraire les cheveux
        hair_region = source_image[y_min:y_max, x_min:x_max].copy()
        hair_mask_region = source_hair_mask[y_min:y_max, x_min:x_max].copy()
        
        # Calculer le facteur d'échelle basé sur la largeur du visage
        source_hair_width = x_max - x_min
        scale_factor = (target_bounds["width"] * 1.5) / max(source_hair_width, 1)
        
        # Redimensionner les cheveux
        new_width = int(hair_region.shape[1] * scale_factor)
        new_height = int(hair_region.shape[0] * scale_factor)
        
        if new_width < 10 or new_height < 10:
            return target_image
        
        hair_resized = cv2.resize(hair_region, (new_width, new_height))
        mask_resized = cv2.resize(hair_mask_region, (new_width, new_height))
        
        # Position sur la cible (au-dessus du front)
        paste_x = target_bounds["center_x"] - new_width // 2
        paste_y = target_bounds["top"] - int(new_height * 0.7)
        
        # Créer le résultat
        result = target_image.copy()
        
        # Effacer les cheveux actuels (optionnel, améliore le résultat)
        if target_hair_mask is not None:
            # Remplir la zone des cheveux actuels avec la couleur de peau
            skin_color = self._estimate_skin_color(target_image, target_landmarks)
            hair_zone = (target_hair_mask > 0.5).astype(np.uint8)
            result = cv2.inpaint(result, hair_zone, 3, cv2.INPAINT_TELEA)
        
        # Coller les nouveaux cheveux
        result = self._paste_with_mask(result, hair_resized, mask_resized, paste_x, paste_y)
        
        # Blending final pour adoucir
        result = self._blend_edges(result, target_image, paste_x, paste_y, new_width, new_height)
        
        return result
    
    def _estimate_skin_color(self, image, landmarks):
        """Estime la couleur de peau depuis les joues."""
        if landmarks is None:
            return (180, 150, 130)
        
        # Points sur les joues
        cheek_points = [50, 280]  # Joue gauche et droite
        colors = []
        
        for idx in cheek_points:
            if idx < len(landmarks):
                x, y = landmarks[idx]
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    colors.append(image[y, x])
        
        if colors:
            return tuple(np.mean(colors, axis=0).astype(int))
        return (180, 150, 130)
    
    def _paste_with_mask(self, background, foreground, mask, x, y):
        """Colle une image avec masque sur un fond."""
        result = background.copy()
        bg_h, bg_w = background.shape[:2]
        fg_h, fg_w = foreground.shape[:2]
        
        # Calculer les zones de chevauchement
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + fg_w)
        y2 = min(bg_h, y + fg_h)
        
        fx1 = x1 - x
        fy1 = y1 - y
        fx2 = fx1 + (x2 - x1)
        fy2 = fy1 + (y2 - y1)
        
        if x2 <= x1 or y2 <= y1:
            return result
        
        # Appliquer avec le masque
        fg_region = foreground[fy1:fy2, fx1:fx2]
        mask_region = mask[fy1:fy2, fx1:fx2]
        bg_region = result[y1:y2, x1:x2]
        
        # Étendre le masque à 3 canaux
        mask_3ch = np.stack([mask_region] * 3, axis=-1)
        
        # Blending
        blended = (fg_region * mask_3ch + bg_region * (1 - mask_3ch)).astype(np.uint8)
        result[y1:y2, x1:x2] = blended
        
        return result
    
    def _blend_edges(self, result, original, x, y, w, h):
        """Adoucit les bords du transfert."""
        # Créer un masque de transition
        mask = np.zeros(result.shape[:2], dtype=np.float32)
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(result.shape[1], x + w)
        y2 = min(result.shape[0], y + h)
        
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        # Blending doux sur les bords
        mask_3ch = np.stack([mask] * 3, axis=-1)
        blended = (result * mask_3ch + original * (1 - mask_3ch)).astype(np.uint8)
        
        return blended


def transfer_hairstyle(target_bgr, source_path, target_landmarks, bisenet_path="models/bisenet_faceparsing.onnx"):
    """
    Fonction utilitaire pour transférer une coiffure.
    
    Args:
        target_bgr: Image cible (ta photo) en BGR
        source_path: Chemin vers l'image source (coiffure à copier)
        target_landmarks: Liste de tuples (x, y) des landmarks MediaPipe
        bisenet_path: Chemin vers le modèle BiSeNet
    
    Returns:
        Image avec la coiffure transférée
    """
    # Charger l'image source
    source_bgr = cv2.imread(source_path)
    if source_bgr is None:
        print(f"[HairTransfer] Impossible de charger: {source_path}")
        return target_bgr
    
    # Créer le transfert
    transfer = HairTransfer(bisenet_path)
    
    # Appliquer
    result = transfer.transfer(target_bgr, source_bgr, target_landmarks)
    
    return result

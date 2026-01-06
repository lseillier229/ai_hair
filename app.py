# app.py
import os
import io
import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import mediapipe as mp
from hair_transfer import HairTransfer  


if "live_face_shape" not in st.session_state:
    st.session_state.live_face_shape = "unknown"

# ----- (optionnel) ONNX Runtime pour BiSeNet -----
try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    HAS_ORT = False

print("HAS_ORT:", HAS_ORT)

st.set_page_config(page_title="Caméra/Image • Forme du visage + Masque", layout="wide")
st.title("💇 Caméra / Image — Forme du visage + Masque (BiSeNet optionnel)")
st.caption("• Onglet **Caméra** (direct) ou **Image** (uploader). • Face Mesh → forme. • BiSeNet ONNX si présent → masque cheveux, sinon Selfie Segmentation (personne).")

# ====== MediaPipe init ======
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_seg = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Indices utiles Face Mesh (468 pts)
IDX = {
    "chin": 152, "forehead": 10,
    "left_zyg": 93, "right_zyg": 323,
    "left_jaw": 234, "right_jaw": 454,
    "left_temples": 127, "right_temples": 356
}

# ====== Catalogue de coiffures par forme de visage ======
HAIRSTYLE_RECOMMENDATIONS = {
    "oval": {
        "description": "Visage ovale - La forme idéale ! Presque toutes les coiffures vous vont.",
        "styles": [
            {"name": "Pompadour", "description": "Volume sur le dessus, côtés courts", "convient": "⭐⭐⭐"},
            {"name": "Quiff", "description": "Mèche volumineuse vers l'arrière", "convient": "⭐⭐⭐"},
            {"name": "Side Part", "description": "Raie sur le côté, classique et élégant", "convient": "⭐⭐⭐"},
            {"name": "Buzz Cut", "description": "Coupe très courte uniforme", "convient": "⭐⭐⭐"},
            {"name": "Cheveux longs", "description": "Mi-longs ou longs, toutes textures", "convient": "⭐⭐⭐"},
        ]
    },
    "round": {
        "description": "Visage rond - Privilégiez le volume en hauteur pour allonger.",
        "styles": [
            {"name": "Pompadour haut", "description": "Maximum de volume sur le dessus", "convient": "⭐⭐⭐"},
            {"name": "Quiff texturé", "description": "Hauteur et texture pour allonger", "convient": "⭐⭐⭐"},
            {"name": "Faux Hawk", "description": "Crête douce, côtés courts", "convient": "⭐⭐⭐"},
            {"name": "Side Part haut", "description": "Raie avec volume", "convient": "⭐⭐"},
            {"name": "Éviter", "description": "Coupes trop plates ou franges droites", "convient": "❌"},
        ]
    },
    "square": {
        "description": "Visage carré - Mâchoire forte, adoucissez les angles.",
        "styles": [
            {"name": "Textured Crop", "description": "Coupe courte texturée, naturelle", "convient": "⭐⭐⭐"},
            {"name": "Side Part souple", "description": "Raie décontractée, pas trop stricte", "convient": "⭐⭐⭐"},
            {"name": "Cheveux mi-longs", "description": "Adoucit les angles de la mâchoire", "convient": "⭐⭐⭐"},
            {"name": "Fringe (frange)", "description": "Frange texturée sur le front", "convient": "⭐⭐"},
            {"name": "Éviter", "description": "Coupes trop géométriques ou flat top", "convient": "❌"},
        ]
    },
    "heart": {
        "description": "Visage cœur - Front large, menton fin. Équilibrez les proportions.",
        "styles": [
            {"name": "Side Part moyen", "description": "Volume latéral pour équilibrer", "convient": "⭐⭐⭐"},
            {"name": "Fringe latérale", "description": "Frange sur le côté, réduit le front", "convient": "⭐⭐⭐"},
            {"name": "Textured medium", "description": "Mi-long avec texture", "convient": "⭐⭐⭐"},
            {"name": "Cheveux longs", "description": "Ajoute de volume près du menton", "convient": "⭐⭐"},
            {"name": "Éviter", "description": "Trop de volume sur le dessus", "convient": "❌"},
        ]
    },
    "diamond": {
        "description": "Visage diamant - Pommettes larges, front et menton étroits.",
        "styles": [
            {"name": "Fringe texturée", "description": "Élargit visuellement le front", "convient": "⭐⭐⭐"},
            {"name": "Side Swept", "description": "Balayé sur le côté, volume haut", "convient": "⭐⭐⭐"},
            {"name": "Textured Crop", "description": "Court et texturé", "convient": "⭐⭐⭐"},
            {"name": "Cheveux mi-longs", "description": "Équilibre les pommettes", "convient": "⭐⭐"},
            {"name": "Éviter", "description": "Côtés trop volumineux", "convient": "❌"},
        ]
    },
    "oblong": {
        "description": "Visage oblong/allongé - Ajoutez de la largeur, réduisez la longueur.",
        "styles": [
            {"name": "Fringe épaisse", "description": "Réduit visuellement la longueur", "convient": "⭐⭐⭐"},
            {"name": "Textured Crop court", "description": "Pas trop de hauteur", "convient": "⭐⭐⭐"},
            {"name": "Side Part bas", "description": "Volume latéral, pas en hauteur", "convient": "⭐⭐⭐"},
            {"name": "Waves/Boucles", "description": "Ajoute de la largeur", "convient": "⭐⭐⭐"},
            {"name": "Éviter", "description": "Pompadour haut, coiffures qui allongent", "convient": "❌"},
        ]
    },
    "unknown": {
        "description": "Forme non détectée - Voici des styles polyvalents.",
        "styles": [
            {"name": "Textured Crop", "description": "Coupe passe-partout", "convient": "⭐⭐"},
            {"name": "Side Part classique", "description": "Toujours élégant", "convient": "⭐⭐"},
            {"name": "Buzz Cut", "description": "Simple et efficace", "convient": "⭐⭐"},
        ]
    }
}

def get_recommendations(face_shape):
    return HAIRSTYLE_RECOMMENDATIONS.get(face_shape, HAIRSTYLE_RECOMMENDATIONS["unknown"])

def dist(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def face_shape_from_landmarks(landmarks_xy):
    pts = {k: landmarks_xy[v] for k, v in IDX.items()}
    length = dist(pts["forehead"], pts["chin"])
    width_cheek = dist(pts["left_zyg"], pts["right_zyg"])
    width_jaw = dist(pts["left_jaw"], pts["right_jaw"])
    width_temples = dist(pts["left_temples"], pts["right_temples"])
    ratios = {
        "cheek_to_length": width_cheek / (length + 1e-6),
        "jaw_to_cheek": width_jaw / (width_cheek + 1e-6),
        "temples_to_cheek": width_temples / (width_cheek + 1e-6),
    }
    cheek_len = ratios["cheek_to_length"]; jaw_cheek = ratios["jaw_to_cheek"]; temp_cheek = ratios["temples_to_cheek"]
    shape = "oval"
    if cheek_len > 0.95: shape = "round"
    if cheek_len <= 0.80: shape = "oblong"
    if (jaw_cheek > 0.98) and (temp_cheek > 0.98) and (0.80 <= cheek_len <= 0.95): shape = "square"
    if (jaw_cheek < 0.92) and (temp_cheek <= 0.98) and (0.80 <= cheek_len <= 0.95): shape = "heart" if temp_cheek < 0.94 else "diamond"
    return shape, ratios

# ====== BiSeNet ONNX helper ======
class BiseNetHairONNX:
    def __init__(self, onnx_path: str, hair_class_id: int = 13, providers=None):
        assert HAS_ORT, "onnxruntime non installé"
        if providers is None:
            providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = max(1, os.cpu_count()//2)

        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        try:
            h, w = [d if isinstance(d, int) else 256 for d in inp.shape[-2:]]
            self.input_size = int(h)
        except Exception:
            self.input_size = 256

        outs = self.session.get_outputs()
        self.output_name = max(outs, key=lambda o: len([d for d in (o.shape or []) if d is not None])).name
        self.hair_id = hair_class_id

    def _pre(self, bgr, mode="tanh"):
        img = cv2.resize(bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        if mode == "tanh":
            img = (img - 0.5)/0.5
        elif mode == "imagenet":
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean)/std
        img = np.transpose(img, (2,0,1))[None,...].astype(np.float32)
        return img

    def infer_hair(self, bgr, out_h, out_w, apply_softmax=False, pre_mode="tanh"):
        x = self._pre(bgr, mode=pre_mode)
        y = self.session.run([self.output_name], {self.input_name: x})[0]

        if y.ndim == 4:
            logits = y
            if apply_softmax:
                logits = logits - logits.max(axis=1, keepdims=True)
                probs = np.exp(logits); probs /= (probs.sum(axis=1, keepdims=True) + 1e-8)
                pred = np.argmax(probs, axis=1)[0].astype(np.uint8)
            else:
                pred = np.argmax(logits, axis=1)[0].astype(np.uint8)
        elif y.ndim == 3:
            pred = y[0].astype(np.uint8)
        else:
            raise RuntimeError(f"Sortie ONNX inattendue: shape={y.shape}")

        hair = (pred == self.hair_id).astype(np.float32)
        hair = cv2.GaussianBlur(hair, (5,5), 0)
        hair = cv2.resize(hair, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(hair, 0.0, 1.0)


def overlay_mask(frame_bgr, mask_prob, alpha=0.35, color=(255,0,0)):
    if alpha <= 0.0:
        return frame_bgr
    m = (mask_prob*255).astype(np.uint8)
    m = cv2.GaussianBlur(m, (9,9), 0)
    over = frame_bgr.copy()
    colored = np.zeros_like(frame_bgr); colored[:] = color
    over[m>128] = cv2.addWeighted(frame_bgr[m>128], 1-alpha, colored[m>128], alpha, 0)
    return over

# ====== Filtre coiffure rapide pour le live ======
class FastHairFilter:
    """
    Filtre 'Snap' rapide :
    - segmente les cheveux de la source UNE FOIS avec BiSeNet
    - ensuite, par frame : juste resize + collage sur le visage
    """
    def __init__(self, source_bgr, onnx_path, hair_class_id=13):
        self.ready = False
        self.hair_region = None
        self.hair_mask_region = None

        if not HAS_ORT or not os.path.isfile(onnx_path):
            print("[FastHairFilter] onnxruntime ou modèle introuvable, désactivé.")
            return

        try:
            bisenet = BiseNetHairONNX(onnx_path, hair_class_id=hair_class_id)
            sh, sw = source_bgr.shape[:2]
            hair_mask = bisenet.infer_hair(source_bgr, sh, sw)
            ys, xs = np.where(hair_mask > 0.5)
            if len(ys) == 0:
                print("[FastHairFilter] Aucun cheveux trouvé dans la source.")
                return

            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            self.hair_region = source_bgr[y_min:y_max, x_min:x_max].copy()
            self.hair_mask_region = hair_mask[y_min:y_max, x_min:x_max].copy().astype(np.float32)
            self.ready = True
            print("[FastHairFilter] Coiffure prête pour le live.")

        except Exception as e:
            print(f"[FastHairFilter] Erreur init: {e}")
            self.ready = False

    @staticmethod
    def _get_face_bounds(landmarks):
        if landmarks is None or len(landmarks) < 468:
            return None
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

    def apply(self, target_bgr, landmarks_xy, hairline_ratio=0.45):
        """
        hairline_ratio = part de la hauteur de la coiffure au-dessus du front.
        0.25 = trop bas (sur le front), 0.45 ≈ plus au niveau de la racine.
        """
        if not self.ready or self.hair_region is None or self.hair_mask_region is None:
            return target_bgr
        if landmarks_xy is None:
            return target_bgr

        bounds = self._get_face_bounds(landmarks_xy)
        if bounds is None:
            return target_bgr

        th, tw = target_bgr.shape[:2]
        src_h, src_w = self.hair_region.shape[:2]

        face_w = bounds["width"]
        face_h = bounds["height"]

        scale_w = (face_w * 1.5) / max(src_w, 1)
        scale_h = (face_h * 1.4) / max(src_h, 1)
        scale = min(scale_w, scale_h)

        new_w = max(10, int(src_w * scale))
        new_h = max(10, int(src_h * scale))

        hair_resized = cv2.resize(self.hair_region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(self.hair_mask_region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # ICI on remonte un peu plus la coiffure
        paste_x = int(bounds["center_x"] - new_w // 2)
        paste_y = int(bounds["top"] - new_h * hairline_ratio)

        x1 = max(0, paste_x)
        y1 = max(0, paste_y)
        x2 = min(tw, paste_x + new_w)
        y2 = min(th, paste_y + new_h)

        if x2 <= x1 or y2 <= y1:
            return target_bgr

        fx1 = x1 - paste_x
        fy1 = y1 - paste_y
        fx2 = fx1 + (x2 - x1)
        fy2 = fy1 + (y2 - y1)

        fg = hair_resized[fy1:fy2, fx1:fx2].astype(np.float32)
        mk = mask_resized[fy1:fy2, fx1:fx2].astype(np.float32)[..., None]
        mk_3 = np.repeat(mk, 3, axis=2)

        out = target_bgr.copy().astype(np.float32)
        bg_roi = out[y1:y2, x1:x2]

        blended = fg * mk_3 + bg_roi * (1.0 - mk_3)
        out[y1:y2, x1:x2] = blended

        return out.astype(np.uint8)

# ====== UI (contrôles) ======
st.sidebar.header("Modes")
mode = st.sidebar.radio("Choisis un mode :", ["🖼️ Image (upload)", "🎥 Caméra (live)"], index=0)

use_bisenet = st.sidebar.checkbox("Utiliser BiSeNet ONNX si disponible (masque)", value=True)
bisenet_path = st.sidebar.text_input("Chemin modèle BiSeNet (.onnx)", "models/bisenet_faceparsing.onnx")

st.sidebar.markdown("---")
show_mask = st.sidebar.checkbox("Afficher l'overlay du masque", value=True)
mask_color_choice = st.sidebar.selectbox("Couleur du masque", ["Rouge","Bleu","Vert","Jaune"], index=0)
color_map = {"Rouge": (0,0,255), "Bleu": (255,0,0), "Vert": (0,255,0), "Jaune": (0,255,255)}
mask_color = color_map[mask_color_choice]
mask_alpha = 0.35 if show_mask else 0.0

st.sidebar.markdown("---")
st.sidebar.subheader("Filtre coiffure (live)")
auto_hair = st.sidebar.checkbox("🎯 Auto: recommander + appliquer une coiffure", value=True)
enable_hair_filter = st.sidebar.checkbox("Activer le filtre coiffure sur la caméra", value=False, disabled=auto_hair)

hair_source_path = st.sidebar.text_input(
    "Chemin de l'image coiffure à tester (mode manuel)",
    value="hairstyle_images/img_align_celeba/000005.jpg",
    disabled=auto_hair
)


# ====== MODE CAMÉRA ======
def pick_hairstyle_source_for_shape(face_shape: str, index_path="hairstyle_index"):
    """
    Retourne un chemin d'image coiffure (source) à appliquer en live.
    - Si l'index FAISS existe : on prend une reco IA (1ère)
    - Sinon : fallback sur des images "demo" (à adapter)
    """
    index_ok = os.path.exists(os.path.join(index_path, "faiss.index"))

    if index_ok:
        try:
            from hairstyle_search import recommend_for_face_shape
            results = recommend_for_face_shape(face_shape, current_image=None, top_k=1, index_path=index_path)
            if results:
                img_path, score, style_name = results[0]
                # normaliser le chemin
                return os.path.normpath(img_path).replace("\\", "/")
        except Exception as e:
            print("[AutoHair] IA indisponible, fallback:", e)

    # ===== Fallback si pas d'index =====
    fallback = {
        "round":  "hairstyle_images/img_align_celeba/000005.jpg",
        "oval":   "hairstyle_images/img_align_celeba/000010.jpg",
        "square": "hairstyle_images/img_align_celeba/000020.jpg",
        "heart":  "hairstyle_images/img_align_celeba/000030.jpg",
        "diamond":"hairstyle_images/img_align_celeba/000040.jpg",
        "oblong": "hairstyle_images/img_align_celeba/000050.jpg",
        "unknown":"hairstyle_images/img_align_celeba/000005.jpg",
    }
    return fallback.get(face_shape, fallback["unknown"])



class LiveTransformer(VideoTransformerBase):
    def __init__(self, use_bisenet=False, bisenet_path="models/bisenet_faceparsing.onnx",
                 mask_alpha=0.35, mask_color=(255,0,0),
                 enable_hair_filter=False, hair_source_path=None):

        self.frame_idx = 0
        self.last_shape = "oval"
        self.last_landmarks = None

        # Face mesh SANS refine_landmarks pour gagner du temps
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.seg_fallback = mp_selfie_seg.SelfieSegmentation(model_selection=1)
        self.bisenet_path = bisenet_path

        self.use_bisenet = False
        self.bisenet = None
        self.mask_alpha = mask_alpha
        self.mask_color = mask_color
        self.auto_hair = auto_hair
        self.current_hair_source = None
        self.stable_shape = None
        self.stable_count = 0

        if use_bisenet and HAS_ORT and os.path.isfile(bisenet_path):
            try:
                self.bisenet = BiseNetHairONNX(bisenet_path, hair_class_id=13)
                self.use_bisenet = True
                print("[BiSeNet] prêt pour le masque live.")
            except Exception as e:
                print("[BiSeNet] Load failed:", e)

        self.enable_hair_filter = enable_hair_filter or self.auto_hair
        self.fast_hair = None

        if (not self.auto_hair) and self.enable_hair_filter and hair_source_path and os.path.isfile(hair_source_path):
            try:
                src_bgr = cv2.imread(hair_source_path)
                if src_bgr is not None:
                    self.fast_hair = FastHairFilter(src_bgr, bisenet_path)
                    self.current_hair_source = hair_source_path
            except Exception as e:
                print(f"[Live] Erreur init FastHairFilter: {e}")


    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_full = frame.to_ndarray(format="bgr24")
        h_full, w_full, _ = img_full.shape

        # Downscale pour traitement
        target_width = 480
        scale = target_width / w_full
        new_w = target_width
        new_h = int(h_full * scale)

        img = cv2.resize(img_full, (new_w, new_h))
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # FaceMesh avec frame skipping
        self.frame_idx += 1
        recompute_face = (self.frame_idx % 3 == 0)

        landmarks_xy = None
        shape = self.last_shape
        if self.stable_shape == shape:
            self.stable_count += 1
        else:
            self.stable_shape = shape
            self.stable_count = 1

        # Après 4 validations (donc ~ 4 * 3 frames), on considère la forme stable
        if self.auto_hair and self.stable_count == 4:
            new_source = pick_hairstyle_source_for_shape(shape)
            if new_source and new_source != self.current_hair_source and os.path.isfile(new_source):
                try:
                    src_bgr = cv2.imread(new_source)
                    if src_bgr is not None:
                        self.fast_hair = FastHairFilter(src_bgr, self.bisenet_path)
                        self.current_hair_source = new_source
                        print(f"[AutoHair] New hair source: {new_source}")
                except Exception as e:
                    print("[AutoHair] Erreur chargement:", e)
        if recompute_face:
            res = self.face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0]
                landmarks_xy = [(int(lm.x * w), int(lm.y * h)) for lm in lms.landmark]
                pts = np.array(landmarks_xy, dtype=np.float32)
                shape, _ = face_shape_from_landmarks(pts)
                self.last_shape = shape
                st.session_state.live_face_shape = shape

                self.last_landmarks = landmarks_xy
                # ❌ On NE dessine PAS les traits blancs en live
        else:
            landmarks_xy = self.last_landmarks

        out = img.copy()
        label = "pas bisenet"

        # Filtre coiffure
        if self.enable_hair_filter and self.fast_hair is not None and landmarks_xy is not None:
            out = self.fast_hair.apply(out, landmarks_xy, hairline_ratio=0.45)
            label = "filtre coiffure"

        # Sinon masque rouge
        if label != "filtre coiffure":
            if self.use_bisenet and self.bisenet is not None:
                try:
                    hair = self.bisenet.infer_hair(img, h, w)
                    out = overlay_mask(out, hair, alpha=self.mask_alpha, color=self.mask_color)
                    label = "cheveux (BiSeNet)"
                except Exception as e:
                    print(f"[BiSeNet] Erreur, fallback selfie_seg: {e}")
                    m = self.seg_fallback.process(rgb).segmentation_mask
                    out = overlay_mask(out, m, alpha=self.mask_alpha, color=self.mask_color)
            else:
                m = self.seg_fallback.process(rgb).segmentation_mask
                out = overlay_mask(out, m, alpha=self.mask_alpha, color=self.mask_color)

        # HUD + upscale
        cv2.rectangle(out, (10,10), (620,90), (0,0,0), -1)
        cv2.putText(
            out,
            f"Forme: {shape}  |  Mode: {label}",
            (20,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2,
            cv2.LINE_AA
        )

        out_full = cv2.resize(out, (w_full, h_full))
        return av.VideoFrame.from_ndarray(out_full, format="bgr24")

# ====== MODE CAMÉRA ======
if mode == "🎥 Caméra (live)":
    st.info("Autorise la caméra dans le navigateur (icône 🔒 à gauche de l'URL). En hébergé, utilise HTTPS.")
    rtc_config = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
    webrtc_streamer(
        key="live-shape-mask",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=lambda: LiveTransformer(
            use_bisenet=use_bisenet,
            bisenet_path=bisenet_path,
            mask_alpha=mask_alpha,
            mask_color=mask_color,
            enable_hair_filter=enable_hair_filter,
            hair_source_path=hair_source_path,
        ),
    )

# ====== MODE IMAGE (upload) ======
# (inchangé, je le laisse comme tu l’avais, avec les analyses, recommandations, etc.)
# Si tu veux aussi retirer les traits blancs sur les images, il suffit de commenter
# le mp_drawing.draw_landmarks dans la partie "Image (upload)" également.

if mode == "🖼️ Image (upload)":
    uploaded = st.file_uploader("Choisir une image", type=["jpg","jpeg","png"])
    if uploaded is None:
        st.warning("Sélectionne une image pour lancer l'analyse.")
    else:
        try:
            file_bytes = uploaded.getvalue()
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            rgb_np = np.array(pil_img)
            bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Impossible de lire l'image: {e}")
            st.stop()

        h, w = bgr.shape[:2]

        # Face mesh statique
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
            res = fm.process(rgb_np)
            shape = "unknown"
            vis = bgr.copy()
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0]
                pts = np.array([(lm.x*w, lm.y*h) for lm in lms.landmark], dtype=np.float32)
                shape, _ = face_shape_from_landmarks(pts)
                mp_drawing.draw_landmarks(
                    image=vis, landmark_list=lms,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

        # Masque (BiSeNet si dispo, sinon fallback personne)
        label = "on a pas bisenet"
        hair_mask = None
        try:
            if use_bisenet and HAS_ORT and os.path.isfile(bisenet_path):
                bn = BiseNetHairONNX(bisenet_path, hair_class_id=13)
                hair_mask = bn.infer_hair(bgr, h, w)
                out = overlay_mask(vis, hair_mask, alpha=mask_alpha, color=mask_color)
                label = "cheveux (BiSeNet)"
            else:
                seg = mp_selfie_seg.SelfieSegmentation(model_selection=1)
                m = seg.process(rgb_np).segmentation_mask
                out = overlay_mask(vis, m, alpha=mask_alpha, color=mask_color)
                hair_mask = m
        except Exception as e:
            st.warning(f"Segmentation fallback (raison: {e})")
            seg = mp_selfie_seg.SelfieSegmentation(model_selection=1)
            m = seg.process(rgb_np).segmentation_mask
            out = overlay_mask(vis, m, alpha=mask_alpha, color=mask_color)
            hair_mask = m

        # ====== ANALYSE DES CHEVEUX ======
        hair_info = {"length": "medium", "texture": "straight", "color": "brown"}
        hair_description = "Analyse non disponible"
        custom_queries = None
        
        try:
            from hair_analysis import analyze_hair, build_clip_query, get_hair_description
            
            landmarks_for_analysis = None
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0]
                landmarks_for_analysis = [(int(lm.x * w), int(lm.y * h)) for lm in lms.landmark]
            
            hair_info = analyze_hair(bgr, hair_mask, landmarks_for_analysis)
            hair_description = get_hair_description(hair_info)
            custom_queries = build_clip_query(shape, hair_info)
            
        except Exception as e:
            print(f"[HairAnalysis] Erreur: {e}")

        cv2.rectangle(out, (10,10), (560,90), (0,0,0), -1)
        cv2.putText(out, f"Forme: {shape}  |  Masque: {label}", (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                 caption=f"Forme: {shape} • Masque: {label}",
                 use_container_width=True)
        
        st.info(f"📊 **Analyse de tes cheveux:** {hair_description}")

        # ====== ESSAYAGE VIRTUEL ======
        st.markdown("---")
        st.subheader("✂️ Essayage virtuel")
        st.caption("Essaie les coiffures recommandées pour ta forme de visage !")
        
        try:
            from virtual_tryon import apply_hairstyle_overlay, setup_demo_overlays, OVERLAY_CATALOG
            setup_demo_overlays()
            
            RECOMMENDED_STYLES = {
                "oval": ["pompadour", "quiff", "side_part", "buzz_cut", "long_straight"],
                "round": ["pompadour", "quiff"],
                "square": ["side_part", "curly", "long_straight"],
                "heart": ["side_part", "curly", "long_straight"],
                "diamond": ["side_part", "curly"],
                "oblong": ["curly", "side_part"],
                "unknown": ["pompadour", "quiff", "side_part"]
            }
            
            recommended_ids = RECOMMENDED_STYLES.get(shape, RECOMMENDED_STYLES["oval"])
            available_overlays = [
                {"id": k, "name": v["name"], "description": v["description"]}
                for k, v in OVERLAY_CATALOG.items()
                if k in recommended_ids
            ]
            
            if not available_overlays:
                st.warning("Aucune coiffure disponible pour l'essayage.")
            else:
                st.info(f"🎯 Coiffures recommandées pour ton visage **{shape}**")
                
                selected_style = st.selectbox(
                    "Choisis une coiffure à essayer :",
                    options=[o["id"] for o in available_overlays],
                    format_func=lambda x: next(o["name"] + " - " + o["description"] for o in available_overlays if o["id"] == x)
                )
                
                hair_colors = {
                    "Naturel": None,
                    "Noir": (20, 20, 20),
                    "Brun": (50, 40, 30),
                    "Châtain": (80, 60, 45),
                    "Blond": (180, 190, 200),
                    "Roux": (60, 80, 180),
                    "Gris": (130, 130, 130)
                }
                selected_color = st.selectbox("Couleur des cheveux :", list(hair_colors.keys()))
                
                if st.button("👁️ Voir le résultat"):
                    if res.multi_face_landmarks:
                        lms = res.multi_face_landmarks[0]
                        landmarks_xy = [(int(lm.x * w), int(lm.y * h)) for lm in lms.landmark]
                        
                        result_img = apply_hairstyle_overlay(
                            bgr.copy(),
                            landmarks_xy,
                            selected_style,
                            hair_color=hair_colors[selected_color]
                        )
                        
                        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                                 caption=f"Essayage: {selected_style}",
                                 use_container_width=True)
                    else:
                        st.error("Visage non détecté. Essaie avec une autre photo.")
                        
        except ImportError as e:
            st.error(f"Module virtual_tryon non trouvé: {e}")

        # ====== RECOMMANDATIONS DE COIFFURES ======
        st.markdown("---")
        st.subheader("💇 Recommandations de coiffures")
        
        reco = get_recommendations(shape)
        st.info(reco["description"])
        
        cols = st.columns(2)
        for i, style in enumerate(reco["styles"]):
            with cols[i % 2]:
                if "Éviter" in style["name"]:
                    st.error(f"**{style['name']}**: {style['description']}")
                else:
                    st.success(f"**{style['name']}** {style['convient']}\n\n{style['description']}")

        # ====== RECOMMANDATIONS IA (AUTOMATIQUE) ======
        st.markdown("---")
        st.subheader("🎯 Nouvelles coiffures recommandées pour toi")
        st.caption("Coiffures DIFFÉRENTES de ton style actuel, mais adaptées à ta forme de visage")
        
        index_exists = os.path.exists("hairstyle_index/faiss.index")
        
        if not index_exists:
            st.warning("⚠️ Index de recherche non configuré. Pour activer les recommandations IA :")
            st.code("pip install torch torchvision transformers faiss-cpu\npython download_dataset.py --sample", language="bash")
        else:
            try:
                from hairstyle_search import recommend_different_styles, search_by_text
                
                with st.spinner("🔄 Analyse de ton style et recherche de nouvelles coiffures..."):
                    results = recommend_different_styles(
                        current_image=rgb_np,
                        face_shape=shape,
                        top_k=6,
                        hair_analysis=hair_info,
                        custom_queries=custom_queries
                    )
                    
                    if results:
                        st.success(f"✅ {len(results)} nouvelles coiffures recommandées pour ton visage {shape}")
                        st.caption("Clique sur 'Essayer' pour voir cette coiffure sur toi !")
                        
                        if "recommended_styles" not in st.session_state:
                            st.session_state.recommended_styles = []
                        st.session_state.recommended_styles = results
                        
                        result_cols = st.columns(3)
                        for i, (img_path, score, style_name) in enumerate(results):
                            with result_cols[i % 3]:
                                try:
                                    img_path_normalized = os.path.normpath(img_path).replace("\\", "/")
                                    result_img = Image.open(img_path_normalized)
                                    st.image(result_img, use_container_width=True)
                                    st.caption(f"💡 {style_name}")
                                    if st.button(f"✂️ Essayer", key=f"try_{i}"):
                                        st.session_state.selected_hairstyle = img_path_normalized
                                except Exception as e:
                                    st.warning(f"❌ {os.path.basename(img_path)}")
                                    st.caption(f"Erreur: {e}")
                    else:
                        st.warning("Aucune recommandation trouvée. Essaie d'ajouter plus d'images au dataset.")
                
                # Section essayage avec transfert de coiffure (offline)
                if "selected_hairstyle" in st.session_state and st.session_state.selected_hairstyle:
                    st.markdown("---")
                    st.subheader("✨ Résultat de l'essayage")
                    
                    try:
                        from hair_transfer import transfer_hairstyle
                        
                        with st.spinner("🔄 Transfert de la coiffure en cours..."):
                            if res.multi_face_landmarks:
                                lms = res.multi_face_landmarks[0]
                                landmarks_xy = [(int(lm.x * w), int(lm.y * h)) for lm in lms.landmark]
                                
                                transferred = transfer_hairstyle(
                                    bgr.copy(),
                                    st.session_state.selected_hairstyle,
                                    landmarks_xy,
                                    bisenet_path=bisenet_path
                                )
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 
                                             caption="Avant", use_container_width=True)
                                with col2:
                                    st.image(cv2.cvtColor(transferred, cv2.COLOR_BGR2RGB), 
                                             caption="Après", use_container_width=True)
                                
                                if st.button("🔄 Essayer une autre coiffure"):
                                    st.session_state.selected_hairstyle = None
                                    st.rerun()
                            else:
                                st.error("Visage non détecté")
                    except Exception as e:
                        st.error(f"Erreur lors du transfert: {e}")
                        st.session_state.selected_hairstyle = None
                if "text_search_results" not in st.session_state:
                    st.session_state.text_search_results = None
                if "text_search_query" not in st.session_state:
                    st.session_state.text_search_query = ""

                # Option recherche textuelle

                st.markdown("---")
                st.subheader("🔍 Ou recherche par description")

                text_query = st.text_input(
                    "Décris la coiffure recherchée :",
                    placeholder="ex: short curly hair, long straight blonde.",
                    value=st.session_state.text_search_query
                )

                if st.button("🔎 Rechercher", key="do_text_search") and text_query:
                    st.session_state.text_search_query = text_query
                    with st.spinner("Recherche en cours."):
                        st.session_state.text_search_results = search_by_text(text_query, top_k=6)

                # ✅ AFFICHAGE PERSISTANT (même après "Essayer")
                results = st.session_state.text_search_results
                if results:
                    st.success(f"✅ {len(results)} résultats pour '{st.session_state.text_search_query}'")
                    result_cols = st.columns(3)

                    for i, (img_path, score) in enumerate(results):
                        with result_cols[i % 3]:
                            try:
                                img_path_normalized = os.path.normpath(img_path).replace("\\", "/")
                                result_img = Image.open(img_path_normalized)
                                st.image(result_img, caption=f"Score: {score:.2f}", use_container_width=True)

                                if st.button("✂️ Essayer", key=f"text_try_{i}"):
                                    st.session_state.selected_hairstyle = img_path_normalized
                                    # ✅ PAS besoin de st.rerun(); le click provoque déjà un rerun
                            except Exception as e:
                                st.warning(f"❌ {os.path.basename(img_path)}")
                                st.caption(f"Erreur: {e}")
                elif st.session_state.text_search_query:
                    st.info("Aucun résultat (ou recherche pas encore lancée).")

                                
            except ImportError as e:
                st.error(f"Module manquant: {e}")
                st.code("pip install torch torchvision transformers faiss-cpu", language="bash")

st.markdown("---")
st.caption("Décoche **Afficher l'overlay du masque** pour garder le masque en interne (clipping) sans teinter l'image.")

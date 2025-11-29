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

        # ----- (optionnel) ONNX Runtime pour BiSeNet -----
        try:
            import onnxruntime as ort
            HAS_ORT = True
        except Exception:
            HAS_ORT = False
        print(HAS_ORT)
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

        def dist(a, b):
            return float(np.hypot(a[0]-b[0], a[1]-b[1]))

        def face_shape_from_landmarks(landmarks_xy):
            """Heuristique simple pour classer la forme du visage (démo)."""
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

        # ====== BiSeNet ONNX helper (optionnel) ======
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
                # Déduit input_size si le modèle le fixe, sinon 512 par défaut
                try:
                    h, w = [d if isinstance(d, int) else 512 for d in inp.shape[-2:]]
                    self.input_size = int(h)
                except Exception:
                    self.input_size = 512

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

                # y: (N,C,H,W) ou (N,H,W)
                if y.ndim == 4:
                    logits = y
                    if apply_softmax:
                        logits = logits - logits.max(axis=1, keepdims=True)
                        probs = np.exp(logits); probs /= (probs.sum(axis=1, keepdims=True) + 1e-8)
                        pred = np.argmax(probs, axis=1)[0].astype(np.uint8)
                    else:
                        pred = np.argmax(logits, axis=1)[0].astype(np.uint8)
                elif y.ndim == 3:
                    pred = y[0].astype(np.uint8)  # déjà argmax côté modèle
                else:
                    raise RuntimeError(f"Sortie ONNX inattendue: shape={y.shape}")

                hair = (pred == self.hair_id).astype(np.float32)
                hair = cv2.GaussianBlur(hair, (5,5), 0)
                hair = cv2.resize(hair, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                return np.clip(hair, 0.0, 1.0)


        def overlay_mask(frame_bgr, mask_prob, alpha=0.35, color=(255,0,0)):
            """Colore (semi-transparent) les pixels du masque. color en BGR."""
            if alpha <= 0.0:
                return frame_bgr
            m = (mask_prob*255).astype(np.uint8)
            m = cv2.GaussianBlur(m, (9,9), 0)
            over = frame_bgr.copy()
            colored = np.zeros_like(frame_bgr); colored[:] = color
            over[m>128] = cv2.addWeighted(frame_bgr[m>128], 1-alpha, colored[m>128], alpha, 0)
            return over

        # ====== UI (contrôles) ======
        st.sidebar.header("Modes")
        mode = st.sidebar.radio("Choisis un mode :", ["🎥 Caméra (live)", "🖼️ Image (upload)"], index=0)

        use_bisenet = st.sidebar.checkbox("Utiliser BiSeNet ONNX si disponible", value=True)
        bisenet_path = st.sidebar.text_input("Chemin modèle BiSeNet (.onnx)", "models/bisenet_faceparsing.onnx")

        st.sidebar.markdown("---")
        show_mask = st.sidebar.checkbox("Afficher l'overlay du masque", value=True)
        mask_color_choice = st.sidebar.selectbox("Couleur du masque", ["Rouge","Bleu","Vert","Jaune"], index=0)
        color_map = {"Rouge": (0,0,255), "Bleu": (255,0,0), "Vert": (0,255,0), "Jaune": (0,255,255)}
        mask_color = color_map[mask_color_choice]
        mask_alpha = 0.35 if show_mask else 0.0

        # ====== MODE CAMÉRA ======
        class LiveTransformer(VideoTransformerBase):
            def __init__(self, use_bisenet=False, bisenet_path="models/bisenet_faceparsing.onnx",
                         mask_alpha=0.35, mask_color=(255,0,0)):
                self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
                self.seg_fallback = mp_selfie_seg.SelfieSegmentation(model_selection=1)
                self.last_shape = "oval"
                self.use_bisenet = False
                self.bisenet = None
                self.mask_alpha = mask_alpha
                self.mask_color = mask_color
                if use_bisenet and HAS_ORT and os.path.isfile(bisenet_path):
                    try:
                        print("cacaboudin")
                        self.bisenet = BiseNetHairONNX(bisenet_path, hair_class_id=13)
                        self.use_bisenet = True
                    except Exception as e:
                        print("[BiSeNet] Load failed:", e)

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                h, w, _ = img.shape
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Face mesh → forme
                res = self.face_mesh.process(rgb)
                shape = self.last_shape
                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0]
                    pts = np.array([(lm.x*w, lm.y*h) for lm in lms.landmark], dtype=np.float32)
                    shape, _ = face_shape_from_landmarks(pts)
                    mp_drawing.draw_landmarks(
                        image=img, landmark_list=lms,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    self.last_shape = shape

                # Masque cheveux (BiSeNet) ou fallback (personne)
                label = "pas bisenet"
                if self.use_bisenet and self.bisenet is not None:
                    try:
                        hair = self.bisenet.infer_hair(img, h, w)
                        out = overlay_mask(img, hair, alpha=self.mask_alpha, color=self.mask_color)
                        label = "cheveux (BiSeNet)"
                    except Exception:
                        m = self.seg_fallback.process(rgb).segmentation_mask
                        out = overlay_mask(img, m, alpha=self.mask_alpha, color=self.mask_color)
                else:
                    m = self.seg_fallback.process(rgb).segmentation_mask
                    out = overlay_mask(img, m, alpha=self.mask_alpha, color=self.mask_color)

                cv2.rectangle(out, (10,10), (560,90), (0,0,0), -1)
                cv2.putText(out, f"Forme: {shape}  |  Masque: {label}", (20,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                return av.VideoFrame.from_ndarray(out, format="bgr24")

        if mode == "🎥 Caméra (live)":
            st.info("Autorise la caméra dans le navigateur (icône 🔒 à gauche de l’URL). En hébergé, utilise HTTPS.")
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
                ),
            )

        # ====== MODE IMAGE (upload) ======
        if mode == "🖼️ Image (upload)":
            uploaded = st.file_uploader("Choisir une image", type=["jpg","jpeg","png"])
            if uploaded is None:
                st.warning("Sélectionne une image pour lancer l’analyse.")
            else:
                try:
                    file_bytes = uploaded.getvalue()  # robuste au rerun Streamlit
                    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                    rgb_np = np.array(pil_img)                     # (H,W,3) RGB
                    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)  # OpenCV en BGR
                except Exception as e:
                    st.error(f"Impossible de lire l'image: {e}")
                    st.stop()

                h, w = bgr.shape[:2]

                # Face mesh statique
                with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
                    res = fm.process(rgb_np)  # MediaPipe accepte un RGB np.array
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
                try:
                    if use_bisenet and HAS_ORT and os.path.isfile(bisenet_path):
                        print("test")
                        bn = BiseNetHairONNX(bisenet_path, hair_class_id=13)
                        hair = bn.infer_hair(bgr, h, w)                   # 0..1
                        out = overlay_mask(vis, hair, alpha=mask_alpha, color=mask_color)
                        label = "cheveux (BiSeNet)"
                    else:
                        print("caca")
                        seg = mp_selfie_seg.SelfieSegmentation(model_selection=1)
                        m = seg.process(rgb_np).segmentation_mask         # 0..1
                        out = overlay_mask(vis, m, alpha=mask_alpha, color=mask_color)
                except Exception as e:
                    print("prout")
                    st.warning(f"Segmentation fallback (raison: {e})")
                    seg = mp_selfie_seg.SelfieSegmentation(model_selection=1)
                    m = seg.process(rgb_np).segmentation_mask
                    out = overlay_mask(vis, m, alpha=mask_alpha, color=mask_color)

                cv2.rectangle(out, (10,10), (560,90), (0,0,0), -1)
                cv2.putText(out, f"Forme: {shape}  |  Masque: {label}", (20,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                         caption=f"Forme: {shape} • Masque: {label}",
                         use_column_width=True)

        st.markdown("---")
        st.caption("Décoche **Afficher l'overlay du masque** pour garder le masque en interne (clipping) sans teinter l'image.")

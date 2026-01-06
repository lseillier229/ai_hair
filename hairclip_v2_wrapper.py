
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAIRCLIPV2_DIR = os.path.join(BASE_DIR, "hairclipv2")

# On fait comme si on lançait le script depuis le repo HairCLIPv2
if HAIRCLIPV2_DIR not in sys.path:
    sys.path.insert(0, HAIRCLIPV2_DIR)
import uuid

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from scripts.Embedding import Embedding
from scripts.text_proxy import TextProxy
from scripts.ref_proxy import RefProxy
from scripts.sketch_proxy import SketchProxy
from scripts.bald_proxy import BaldProxy
from scripts.color_proxy import ColorProxy
from scripts.feature_blending import hairstyle_feature_blending
from utils.seg_utils import vis_seg
from utils.mask_ui import painting_mask
from utils.image_utils import display_image_list, process_display_input
from utils.model_utils import load_base_models
from utils.options import Options


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class HairCLIPV2:
    """
    Wrapper simple autour du pipeline de hairclip_v2_demo.py
    - input : image PIL + texte (ex: 'bowl cut hairstyle') et/ou texte couleur (ex: 'red hair')
    - output : image PIL coiffée
    """

    def __init__(self):
        # Options (comme dans hairclip_v2_demo)
        self.opts = Options().parse(jupyter=True)

        # Répertoires des images et des latents (définis dans Options)
        self.src_img_dir = self.opts.src_img_dir
        self.src_latent_dir = self.opts.src_latent_dir

        os.makedirs(self.src_img_dir, exist_ok=True)
        os.makedirs(self.src_latent_dir, exist_ok=True)

        # Transform comme dans le demo
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

        # Charger les modèles de base (StyleGAN, segmentation, etc.)
        self.g_ema, self.mean_latent_code, self.seg = load_base_models(self.opts)

        # Embedding (inversion)
        self.ii2s = Embedding(self.opts, self.g_ema, self.mean_latent_code[0, 0])

        # Proxies (comme dans le demo)
        self.bald_proxy = BaldProxy(self.g_ema, self.opts.bald_path)
        self.text_proxy = TextProxy(self.opts, self.g_ema, self.seg, self.mean_latent_code)
        self.ref_proxy = RefProxy(self.opts, self.g_ema, self.seg, self.ii2s)
        self.sketch_proxy = SketchProxy(self.g_ema, self.mean_latent_code, self.opts.sketch_path)
        self.color_proxy = ColorProxy(self.opts, self.g_ema, self.seg)

        print("[HairCLIPV2] Modèles chargés.")

    # --- utilitaire : tensor -> PIL ---
    @staticmethod
    def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
        """
        img_tensor: (1,3,H,W) ou (3,H,W) dans l'espace [-1,1] ou normalisé 0.5/0.5
        """
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]

        # on suppose qu'on est dans [-1,1] après le réseau
        img = img_tensor.detach().cpu()
        # ramener sur [0,1]
        if img.min() < 0.0:
            img = (img * 0.5 + 0.5)
        img = img.clamp(0.0, 1.0)
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)

    # --- 1) inversion de l'image en latent + mask cheveux ---
    def _prepare_source(self, pil_img: Image.Image):
        """
        - Sauve l'image sur disque (comme le notebook)
        - Inverse en latent W+ + F
        - Calcule le masque de segmentation
        Retourne: src_latent, src_feature, src_image_tensor, input_mask
        """
        # nom unique pour cette image
        src_name = f"streamlit_{uuid.uuid4().hex[:8]}"
        img_path = os.path.join(self.src_img_dir, f"{src_name}.jpg")

        pil_img.save(img_path)

        # Inversion (comme dans hairclip_v2_demo.py)
        inverted_latent_w_plus, inverted_latent_F = self.ii2s.invert_image_in_FS(image_path=img_path)

        src_latent = inverted_latent_w_plus.to(DEVICE)
        src_feature = inverted_latent_F.to(DEVICE)

        # Image normalisée ([-1,1]) pour la segmentation
        src_image = self.image_transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)

        # Masque : argmax sur la sortie seg[1]
        with torch.no_grad():
            seg_logits = self.seg(src_image)[1]
            input_mask = torch.argmax(seg_logits, dim=1).long().clone().detach()

        return src_latent, src_feature, src_image, input_mask

    # --- 2) editing coiffure (texte ou ref) ---
    def _hairstyle_edit(
        self,
        src_latent,
        src_feature,
        src_image,
        input_mask,
        hairstyle_text: str | None = None,
        hairstyle_ref_path: str | None = None,
    ):
        """
        Pipeline hairstyle_editing du demo, simplifié:
        - pas de sketch interactif
        - pas de peinture de masque à la main
        """
        latent_bald = None
        latent_global = None
        latent_local = None
        local_blending_mask = None
        painted_mask = None  # on ne gère pas le painting_mask interactif ici

        # Si on a une condition globale (texte ou ref), on calcule les latents
        global_cond = None
        if hairstyle_ref_path is not None:
            global_cond = hairstyle_ref_path
        elif hairstyle_text:
            global_cond = hairstyle_text

        if global_cond is not None:
            # latent chauve
            latent_bald, _ = self.bald_proxy(src_latent)

            # ref ou texte
            if isinstance(global_cond, str) and (global_cond.endswith(".jpg") or global_cond.endswith(".png")):
                latent_global, _ = self.ref_proxy(global_cond, src_image, painted_mask=painted_mask)
            else:
                latent_global, _ = self.text_proxy(global_cond, src_image, from_mean=True, painted_mask=painted_mask)

        # blending (comme dans le demo)
        new_src_feature, edited_hairstyle_img = hairstyle_feature_blending(
            self.g_ema,
            self.seg,
            src_latent,
            src_feature,
            input_mask,
            latent_bald=latent_bald,
            latent_global=latent_global,
            latent_local=latent_local,
            local_blending_mask=local_blending_mask,
        )

        return new_src_feature, edited_hairstyle_img

    # --- 3) color editing optionnel ---
    def _color_edit(self, color_cond, edited_hairstyle_img, src_latent, src_feature):
        """
        color_cond: str ou tuple RGB, comme dans le demo:
        - 'red hair' (texte)
        - '108157.jpg' (ref image)
        - (220,220,220) (RGB)
        """
        visual_color_list, visual_final_list = self.color_proxy(
            color_cond, edited_hairstyle_img, src_latent, src_feature
        )
        # on prend la dernière image (résultat final)
        final_img = visual_final_list[-1]
        return final_img

    # --- API principale à appeler depuis Streamlit ---
    def run(
        self,
        src_image: Image.Image,
        hairstyle_text: str | None = None,
        color_cond: str | tuple | None = None,
        hairstyle_ref_filename: str | None = None,
    ) -> Image.Image:
        """
        src_image : PIL.Image (photo uploadée)
        hairstyle_text : ex. 'bowl cut hairstyle', 'curly bob hairstyle'
        color_cond : ex. 'red hair' ou '108157.jpg'
        hairstyle_ref_filename : nom de fichier dans opts.ref_img_dir (optionnel)
        """
        # 1) inversion + mask
        src_latent, src_feature, src_tensor, input_mask = self._prepare_source(src_image)

        # 2) édition coiffure (texte ou ref)
        src_feature_after, edited_hairstyle_img = self._hairstyle_edit(
            src_latent,
            src_feature,
            src_tensor,
            input_mask,
            hairstyle_text=hairstyle_text,
            hairstyle_ref_path=hairstyle_ref_filename,
        )

        # 3) édition couleur optionnelle
        if color_cond is not None:
            final_tensor = self._color_edit(color_cond, edited_hairstyle_img, src_latent, src_feature_after)
        else:
            final_tensor = edited_hairstyle_img

        # 4) tensor -> PIL.Image
        return self._tensor_to_pil(final_tensor)

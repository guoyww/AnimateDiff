import torch
import numpy as np
import cv2


import os
from PIL import Image, ImageDraw
from .n3rMotionPose_tools import save_debug_mask, feather_inside_strict, feather_mask, feather_mask_fast, feather_outside_only, feather_inside,feather_inside_strict, feather_outside_only_alpha, feather_inside_strict2, feather_outside_only_alpha2, save_debug_mask_scale


def ensure_2d(x):
    """
    Force shape [2] (x,y)
    """
    if x is None:
        return None
    if x.ndim == 2:
        x = x[0]
    return x[:2]

#---------------------------------- CLASS POSE ------------------------------------------------------
class Pose:
    def __init__(self, keypoints: torch.Tensor):
        """
        keypoints : [B, 17, 3]  -> normalized [0,1] + confidence
        """
        self.keypoints = keypoints.clone()
        self.B = keypoints.shape[0]
        self.delta = None
        self.angles = None
        self.device = keypoints.device
        self._prev_facial_points = None
        # Ajustement
        self.torce_expand_w=1.6
        self.torce_shrink_h=1.2
        self.bouche_expand_h=0.25
        self.bouche_expand_w=0.60

        # -----------------------------
        # Mapping global des points faciaux
        # -----------------------------

        self.FACIAL_POINT_IDX = {
            'nose': 0,  # 'nez' : 0,
            'neck': 1,  # 'cou' : 1,
            'right_shoulder': 2, # 'épaule_droite' : 2,
            'right_elbow': 3, # 'coude_droit' : 3,
            'right_wrist': 4, # 'poignet_droit' : 4,
            'left_shoulder': 5, # 'épaule_gauche' : 5,
            'left_elbow': 6, # 'coude_gauche' : 6,
            'left_wrist': 7, # 'poignet_gauche' : 7,
            'right_hip': 8, # 'hanche_droite' : 8,
            'right_knee': 9, # 9	genou droit (right knee)
            'right_ankle': 10, # 10	cheville droite (right ankle)
            'left_hip': 11,
            'left_knee': 12, # 12	genou gauche (left knee)
            'left_ankle': 13, # 13	cheville gauche (left ankle)
            'right_eye': 14,
            'left_eye': 15,
            'right_ear': 16,
            'left_ear': 17,
            'mouth': 18,  # Centre de la couche mouth_center
            'chin': 21,
            'left_side_neck': 22,
            'right_side_neck': 23,
            'anchor_neck': 24,
            'hair_root': 25,
            'hair_left': 26,
            'hair_right': 27,
            'hair_top': 28,
            'hair_top_left': 29,
            'hair_top_right': 30,

            'left_top_hair1': 31,
            'left_top_hair2': 32,
            'left_top_hair3': 33,

            'right_top_hair1': 34,
            'right_top_hair2': 35,
            'right_top_hair3': 36,


            'mouth_left_ext': 38, # coin extérieur à la bouche
            'mouth_right_ext':39, # coin extérieur à la bouche

            'mouth_left': 40,    # coin gauche des lèvres supérieures upper_ids=(5,6,7,8,9,10),
            'mouth_right': 41,   # coin droit des lèvres supérieures

            'nose_left': 42,    # coin gauche du nez
            'nose_right': 43,   # coin droit du nez
            # =========================
            # 🔥 VIRTUAL / CINEMATIC POINTS
            # =========================
            'mouth_left_c': 48,    # index approximatif coin gauche des lèvres supérieures
            'mouth_right_c': 49,   # index approximatif coin droit des lèvres supérieures
            'mouth_top': 50,     # index approximatif du haut de la bouche (à ajuster selon ton keypoints)
            'mouth_bottom': 51,  # index approximatif du bas de la bouche

            # =========================
            # 🔥 Point réel
            # =========================
            'front_left_1': 52, # front gauche 1
            'front_left_2': 53, # front gauche 2
            'front_m': 54, # front milleu
            'front_right_1': 55, # front droit 1
            'front_right_2': 56, # front droit 2
            # Lèvre du haut
            'mouth_top_mid_r3': 70,
            'mouth_top_mid_r2': 71,
            'mouth_top_mid_r1': 72,
            'mouth_top_mid': 73,
            'mouth_top_mid_l1': 74,
            'mouth_top_mid_l2': 75,
            'mouth_top_mid_l3': 76,
            # Lèvre du bas
            'mouth_bot_mid_r3': 77,
            'mouth_bot_mid_r2': 78,
            'mouth_bot_mid_r1': 79,
            'mouth_bot_mid': 80,
            'mouth_bot_mid_l1': 81,
            'mouth_bot_mid_l2': 82,
            'mouth_bot_mid_l3': 83,

        }

    def get_head_up(self):
        """
        Retourne le vecteur `head_up` basé sur les points du visage et des cheveux.

        """
        hair_left = self.get_point(self.FACIAL_POINT_IDX["left_top_hair3"]) # hair_left
        hair_right = self.get_point(self.FACIAL_POINT_IDX["right_top_hair3"]) #hair_right

        if hair_left is None or hair_right is None:
            return None

        # Déplacer vers CPU si nécessaire
        if isinstance(hair_left, torch.Tensor) and hair_left.is_cuda:
            hair_left = hair_left.cpu().numpy()  # Déplacer sur le CPU
        elif isinstance(hair_left, torch.Tensor):
            hair_left = hair_left.numpy()

        if isinstance(hair_right, torch.Tensor) and hair_right.is_cuda:
            hair_right = hair_right.cpu().numpy()  # Déplacer sur le CPU
        elif isinstance(hair_right, torch.Tensor):
            hair_right = hair_right.numpy()

        # Calcul du centre des cheveux
        hair_center = (hair_left + hair_right) * 0.5

        # Calcul du vecteur head_up
        neck = self.get_point(self.FACIAL_POINT_IDX["neck"])
        nose = self.get_point(self.FACIAL_POINT_IDX["nose"])

        if neck is None or nose is None:
            return None

        # Déplacer vers CPU si nécessaire
        if isinstance(neck, torch.Tensor) and neck.is_cuda:
            neck = neck.cpu().numpy()
        elif isinstance(neck, torch.Tensor):
            neck = neck.numpy()

        if isinstance(nose, torch.Tensor) and nose.is_cuda:
            nose = nose.cpu().numpy()
        elif isinstance(nose, torch.Tensor):
            nose = nose.numpy()

        # Assurez-vous que neck et nose sont des vecteurs 1D avec 2 éléments
        neck = torch.tensor(neck.squeeze(), device=self.device)
        nose = torch.tensor(nose.squeeze(), device=self.device)
        hair_center = torch.tensor(hair_center.squeeze(), device=self.device)

        # Vérification de la forme
        assert neck.shape == torch.Size([2]), f"Neck should be a 2D vector, but got {neck.shape}"
        assert nose.shape == torch.Size([2]), f"Nose should be a 2D vector, but got {nose.shape}"
        assert hair_center.shape == torch.Size([2]), f"Hair center should be a 2D vector, but got {hair_center.shape}"

        # Calcul du vecteur directionnel `head_up`
        # Par exemple, ici head_up est la direction de la ligne reliant le nez et le cou
        head_up = nose - neck  # Vecteur de direction du cou au nez (comme une approximation)
        head_up = head_up / head_up.norm()  # Normalisation pour avoir une direction unitaire

        # =========================
        # HEAD CENTER (ROBUST)
        # =========================
        head_center = (hair_center + nose + neck) / 3.0

        # Vérification de la forme de head_center avant d'accéder aux indices
        if head_center.shape != torch.Size([2]):
            raise ValueError(f"Expected head_center to be 2D vector, but got shape {head_center.shape}")

        cx = head_center[0]
        cy = head_center[1]

        return head_up


    @staticmethod
    def rotate_points_around_z_axis(points, pivot, angle):
        """
        points: [B,N,2]
        pivot : [B,2] ou [B,1,2]
        angle : [B,1]
        """

        B, N, _ = points.shape

        # =========================================================
        # SANITIZE
        # =========================================================
        points = torch.nan_to_num(points, nan=0.0)
        pivot  = torch.nan_to_num(pivot,  nan=0.0)
        angle  = torch.nan_to_num(angle,  nan=0.0)

        # =========================================================
        # 🔥 SHAPE FIX CRITIQUE
        # =========================================================
        angle = angle.view(B, 1, 1)   # 👉 FORCE SHAPE SAFE

        if pivot.ndim == 2:
            pivot = pivot.unsqueeze(1)  # [B,1,2]

        # =========================================================
        # LIMIT
        # =========================================================
        angle = torch.clamp(angle, -0.2, 0.2)

        # =========================================================
        # ROTATION
        # =========================================================
        cos_a = torch.cos(angle)  # [B,1,1]
        sin_a = torch.sin(angle)

        centered = points - pivot  # [B,N,2]

        x = centered[..., 0:1]  # [B,N,1]
        y = centered[..., 1:2]

        x_new = x * cos_a - y * sin_a  # SAFE broadcast
        y_new = x * sin_a + y * cos_a

        rotated = torch.cat([x_new, y_new], dim=-1)

        return rotated + pivot

    def apply_rotation_on_keypoints(kp, state, device):
        # Récupérer les angles de rotation
        angle_x = state["global_angle_x"]
        angle_y = state["global_angle_y"]
        angle_z = state["global_angle_z"]

        # Calculer la matrice de rotation 3D
        R = rotation_matrix_3d(angle_x, angle_y, angle_z, device)

        # Appliquer la rotation à chaque keypoint
        B, N, _ = kp.shape  # B = batch size, N = nombre de keypoints
        kp_3d = kp  # Assurez-vous que vos keypoints sont en 3D, avec la coordonnée z

        # Appliquer la rotation (R est une matrice 3x3, kp est de dimension (B, N, 3))
        kp_rotated = torch.einsum('bnc,cd->bnd', kp_3d, R)

        return kp_rotated


    def apply_rotation_z(self, angle):
        """
        Applique une rotation sur les points du corps (épaules, coudes, poignets) autour du pivot
        calculé à partir des points du torse et du cou.

        angle : angle de rotation en radians
        """
        # Indices des points à faire pivoter (épaules, coudes, poignets, etc.)
        upper_ids = [
            self.FACIAL_POINT_IDX['left_shoulder'],
            self.FACIAL_POINT_IDX['right_shoulder'],
            self.FACIAL_POINT_IDX['left_elbow'],
            self.FACIAL_POINT_IDX['right_elbow'],
            self.FACIAL_POINT_IDX['left_wrist'],
            self.FACIAL_POINT_IDX['right_wrist'],
            self.FACIAL_POINT_IDX['left_hip'],
            self.FACIAL_POINT_IDX['right_hip'],
            self.FACIAL_POINT_IDX['left_knee'],
            self.FACIAL_POINT_IDX['right_knee'],
            self.FACIAL_POINT_IDX['left_ankle'],
            self.FACIAL_POINT_IDX['right_ankle']
        ]

        # Calculer le pivot comme la moyenne des points du torse et du cou (ici entre 'neck' et les épaules)
        # On prend l'index 1 pour le 'neck' et les indices des épaules gauche et droite
        pivot = self.keypoints[:, [self.FACIAL_POINT_IDX['neck'], self.FACIAL_POINT_IDX['left_shoulder'], self.FACIAL_POINT_IDX['right_shoulder']], :2].mean(dim=1, keepdim=True)

        # Adapter la forme de pivot pour qu'il corresponde à celle des points à faire pivoter
        pivot = pivot.expand(-1, len(upper_ids), -1)

        # Appliquer la rotation des points autour du pivot
        self.keypoints[:, upper_ids, :2] = self.rotate_points_around_z_axis(
            self.keypoints[:, upper_ids, :2],
            pivot,
            angle
        )

    def rotate_points_around_pivot(self, points, pivot, angle):
        """
        Applique une rotation autour du pivot pour les points donnés.

        points : [B, N, 2] - Points à faire pivoter (x, y)
        pivot : [B, N, 2] - Pivot autour duquel faire pivoter
        angle : [B, 1] - Angles de rotation en radians (par batch)

        Retourne les points après rotation.
        """

        # Vérification des NaN dans les points et pivots
        if torch.isnan(points).any():
            raise ValueError("Les points contiennent des valeurs NaN")

        if torch.isnan(pivot).any():
            # Si le pivot contient des NaN, on peut choisir un pivot de secours
            print("Avertissement : Le pivot contient des NaN. Utilisation du pivot de secours (centre de l'épaule).")
            pivot = torch.zeros_like(pivot)  # Par exemple, on remplace par un pivot arbitraire (0, 0)

        if torch.isnan(angle).any() or angle.shape != torch.Size([points.shape[0], 1]):
            # Si l'angle est NaN ou a une forme inattendue, on applique une rotation nulle
            print("Avertissement : L'angle est NaN ou a une forme incorrecte. Rotation nulle appliquée.")
            angle = torch.zeros_like(angle)  # Appliquer une rotation nulle

        # Calculer le cosinus et sinus de l'angle
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # Centrer les points autour du pivot
        points_centered = points - pivot

        # Appliquer la rotation
        x_new = points_centered[..., 0] * cos_a - points_centered[..., 1] * sin_a
        y_new = points_centered[..., 0] * sin_a + points_centered[..., 1] * cos_a

        # Revenir aux coordonnées d'origine
        rotated_points = torch.stack([x_new, y_new], dim=-1) + pivot

        # Assurer que les valeurs sont valides (sans NaN)
        rotated_points = torch.nan_to_num(rotated_points, nan=0.0)

        # Retourner les points après rotation
        return rotated_points

    def apply_rotation(self, angle):
        """
        Applique une rotation sur les points du corps (épaules, coudes, poignets) autour du pivot
        calculé à partir des points du torse et du cou.

        angle : angle de rotation en radians
        """
        # Indices des points à faire pivoter (épaules, coudes, poignets)
        upper_ids = [
            self.FACIAL_POINT_IDX['left_shoulder'],
            self.FACIAL_POINT_IDX['right_shoulder'],
            self.FACIAL_POINT_IDX['left_elbow'],
            self.FACIAL_POINT_IDX['right_elbow'],
            self.FACIAL_POINT_IDX['left_wrist'],
            self.FACIAL_POINT_IDX['right_wrist']
        ]

        # Calculer le pivot comme la moyenne des points dans la plage spécifiée (ici de 5 à 12)
        pivot = self.keypoints[:, 5:min(13, self.B), :2].mean(dim=1, keepdim=True)

        # Adapter la forme de pivot pour qu'il corresponde à celle des points à faire pivoter
        pivot = pivot.expand(-1, len(upper_ids), -1)

        # Appliquer la rotation des points autour du pivot
        self.keypoints[:, upper_ids, :2] = self.rotate_points_around_pivot(
            self.keypoints[:, upper_ids, :2],
            pivot,
            angle
        )

    def get_bouche_expand_h(self):
        return self.bouche_expand_h

    def get_bouche_expand_w(self):
        return self.bouche_expand_w

    def compute_pose_context(self, facial_points):
        """
        Returns global pose context for motion modules.
        Stable face + profile + orientation signals.
        """

        ctx = {}

        eps = 1e-6

        # =========================================================
        # SHOULDER AXIS (base body orientation)
        # =========================================================
        if 'left_shoulder' in facial_points and 'right_shoulder' in facial_points:
            ls = facial_points['left_shoulder']
            rs = facial_points['right_shoulder']

            shoulder_vec = rs - ls
            shoulder_width = torch.norm(shoulder_vec, dim=-1, keepdim=True) + eps
            shoulder_dir = shoulder_vec / shoulder_width

            # raw proxy: smaller width => stronger profile (normalized later)
            ctx["shoulder_vec"] = shoulder_vec
            ctx["shoulder_dir"] = shoulder_dir
            ctx["shoulder_width"] = shoulder_width.squeeze(-1)

        else:
            ctx["shoulder_dir"] = None
            ctx["shoulder_width"] = None

        # =========================================================
        # HEAD CENTER (stable anchor)
        # =========================================================
        if 'nose' in facial_points and 'neck' in facial_points:
            ctx["head_center"] = (facial_points['nose'] + facial_points['neck']) * 0.5
        else:
            ctx["head_center"] = None

        # =========================================================
        # FACE WIDTH (eye distance = better than shoulders for pose)
        # =========================================================
        eye_width = None
        if 'left_eye' in facial_points and 'right_eye' in facial_points:
            le = facial_points['left_eye']
            re = facial_points['right_eye']
            eye_width = torch.norm(re - le, dim=-1, keepdim=True)
            ctx["eye_width"] = eye_width

        # =========================================================
        # NOSE OFFSET (very strong profile cue)
        # =========================================================
        nose_offset = None
        if 'nose' in facial_points and 'left_eye' in facial_points and 'right_eye' in facial_points:
            nose = facial_points['nose']
            le = facial_points['left_eye']
            re = facial_points['right_eye']

            eye_mid = (le + re) * 0.5
            nose_offset = torch.norm(nose - eye_mid, dim=-1, keepdim=True)
            ctx["nose_offset"] = nose_offset

        # =========================================================
        # PROFILE FACTOR (robust fusion)
        # =========================================================
        profile_terms = []

        # 1. shoulder compression
        if ctx.get("shoulder_width") is not None:
            # inverse normalized (tune range empirically)
            p_shoulder = torch.exp(-ctx["shoulder_width"])
            profile_terms.append(p_shoulder)

        # 2. nose deviation (strong signal for profile)
        if nose_offset is not None:
            p_nose = torch.tanh(nose_offset * 3.0)
            profile_terms.append(p_nose)

        # 3. eye width consistency (fallback face openness)
        if eye_width is not None:
            p_eye = torch.exp(-eye_width)
            profile_terms.append(p_eye)

        if len(profile_terms) > 0:
            profile = sum(profile_terms) / len(profile_terms)
            profile = torch.clamp(profile, 0.0, 1.0)
        else:
            profile = torch.zeros_like(facial_points['nose'][..., 0])

        ctx["profile_factor"] = profile

        # =========================================================
        # FACE ORIENTATION VECTOR (useful for breathing + mouth)
        # =========================================================
        if ('left_eye' in facial_points and
            'right_eye' in facial_points and
            'nose' in facial_points):

            le = facial_points['left_eye']
            re = facial_points['right_eye']
            nose = facial_points['nose']

            eye_vec = re - le
            eye_mid = (le + re) * 0.5

            face_vec = nose - eye_mid
            face_norm = torch.norm(face_vec, dim=-1, keepdim=True) + eps

            ctx["face_dir"] = face_vec / face_norm
        else:
            ctx["face_dir"] = None

        return ctx
    # -----------------------------
    # 🔹 UPDATE HAUT DU CORPS (smooth)
    # -----------------------------
    def update_upper_body(self, coords_dict, strength=0.35):
        """
        coords_dict : dict avec les points détectés (nose, neck, shoulder, elbow, wrist, clavicle, eyes, ears, mouth)
        strength : interpolation entre l'ancien et le nouveau (0=full old, 1=full new)
        """
        keypoints_np = self.keypoints.clone().cpu().numpy()[0]

        for name, idx in self.FACIAL_POINT_IDX.items():
            if name in coords_dict:
                new_coord = np.array(coords_dict[name], dtype=np.float32)

                # Smooth
                prev_coord = keypoints_np[idx] if self._prev_keypoints is None else self._prev_keypoints[0, idx].cpu().numpy()
                keypoints_np[idx] = prev_coord * (1 - strength) + new_coord * strength

        # Mise à jour du tensor
        keypoints_np = np.expand_dims(keypoints_np, axis=0)
        keypoints_tensor = torch.from_numpy(keypoints_np).to(self.device)

        # Update memory
        self._prev_keypoints = keypoints_tensor.clone()
        self.keypoints = keypoints_tensor.clone()

        return keypoints_tensor

    # -----------------------------
    # 🔹 ROTATION TORSE
    # -----------------------------
    def rotate_torso(self, angle_deg: float, strength: float = 1.0):
        """
        Applique une rotation 2D du torse (hips + épaules) autour du centre du buste.
        angle_deg : rotation en degrés (+ = sens horaire)
        strength  : 0 = pas de rotation, 1 = rotation totale
        """
        if self.B == 0:
            return

        # Points du torse à affecter
        torso_points = ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip']
        idxs = [self.FACIAL_POINT_IDX[p] for p in torso_points]

        # Centre du torse
        center = self.keypoints[:, idxs, :2].mean(dim=1, keepdim=True)  # [B,1,2]

        # Conversion angle -> radians
        angle_rad = angle_deg * (np.pi / 180.0) * strength

        # Matrice de rotation 2D
        rot_matrix = torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad)],
            [torch.sin(angle_rad),  torch.cos(angle_rad)]
        ], device=self.device)

        # Appliquer rotation
        torso_coords = self.keypoints[:, idxs, :2] - center  # centrer
        torso_coords_rot = torch.matmul(torso_coords, rot_matrix.T) + center  # rotation + recentrage

        # Mise à jour keypoints
        self.keypoints[:, idxs, :2] = torso_coords_rot
        return self.keypoints[:, idxs, :2]

    # =========================
    # GETTER
    # =========================

    def get_prev_facial_points(self):
        return self._prev_facial_points

    #self.get_torce_expand_w()
    def get_torce_expand_w(self):
        return self.torce_expand_w

    #self.get_torce_shrink_h()
    def get_torce_shrink_h(self):
        return self.torce_shrink_h

    def has_point(self, name: str) -> bool:
        """
        Vérifie si un point existe dans le mapping ET dans le tensor.
        """
        idx = self.FACIAL_POINT_IDX.get(name, None)
        if idx is None:
            return False
        return idx < self.keypoints.shape[1]

    # =========================
    # SETTER
    # =========================
    def set_prev_facial_points(self, points):
        self._prev_facial_points = points

    def get_center(self):
        # suppose keypoints shape: (B, N, 2) ou (N, 2)
        kpts = self.keypoints

        if kpts.dim() == 3:
            kpts = kpts[0]

        valid = kpts[kpts[..., 0] > 0]  # filtre simple si coords invalides
        if valid.shape[0] == 0:
            return torch.zeros(2, device=kpts.device)

        return valid.mean(dim=0)

    def estimate_facial_points_full(self, smooth=0.8):
        """
        Version complète + head-aware + stable + frame facial cohérent
        """

        points = {}

        kp = self.keypoints
        nose = kp[:, 0, :2]

        hair_root = kp[:, 25, :2]
        hair_left = kp[:, 26, :2]
        hair_right = kp[:, 27, :2]

        hair_top = kp[:, 28, :2]
        left_top_hair3 = kp[:, 33, :2]
        right_top_hair3 = kp[:, 36, :2]


        right_eye = kp[:, 14, :2] if kp.shape[1] > 14 else None
        left_eye  = kp[:, 15, :2] if kp.shape[1] > 15 else None

        right_ear = kp[:, 16, :2] if kp.shape[1] > 16 else None
        left_ear  = kp[:, 17, :2] if kp.shape[1] > 17 else None

        # 'mouth_right': 41,   # coin droit des lèvres supérieures
        mouth_right = kp[:, 41, :2] if kp.shape[1] > 41 else None
        # 'mouth_left': 40,    # coin gauche des lèvres supérieures
        mouth_left  = kp[:, 42, :2] if kp.shape[1] > 42 else None
        # 'mouth': 18, # centre des lèvres supérieures
        mouth_center = kp[:, 18, :2] if kp.shape[1] > 18 else None


        # =========================
        # FALLBACK STABLE DISTANCE
        # =========================
        if right_eye is not None and left_eye is not None:
            eye_center = (left_eye + right_eye) * 0.5
            eye_vec = left_eye - right_eye
            eye_dist = torch.norm(eye_vec, dim=1, keepdim=True).clamp(min=1e-5)
        else:
            eye_center = nose + torch.tensor([0.0, -0.08], device=self.device)
            eye_vec = torch.tensor([1.0, 0.0], device=self.device).repeat(self.B, 1)
            eye_dist = torch.full((self.B, 1), 0.12, device=self.device)

        # =========================
        # HEAD FRAME (IMPORTANT)
        # =========================
        eye_dir = eye_vec / eye_dist

        # vertical axis = nose -> eyes center
        head_up = (eye_center - nose)
        head_up = head_up / (torch.norm(head_up, dim=1, keepdim=True) + 1e-6)

        # perpendicular axis (face width direction)
        head_right = torch.stack([-head_up[:, 1], head_up[:, 0]], dim=1)

        # =========================
        # HEAD CENTER STABLE
        # =========================
        head_center = (nose + eye_center) * 0.5

        # =========================
        # SCALE FACTORS
        # =========================
        face_w = eye_dist
        face_h = eye_dist * 1.4

        # =========================
        # MOUTH (FRAME-BASED)
        # =========================
        if mouth_center is not None :
            mouth_center = head_center + head_up * (0.65 * face_h)

        if mouth_left is not None :
            mouth_left = mouth_center - head_right * (0.35 * face_w)

        if mouth_right is not None :
            mouth_right = mouth_center + head_right * (0.35 * face_w)

        # Calcul de la position du haut de la bouche
        mouth_top = mouth_center - head_up * (0.15 * face_h)
        # Calcul de la position du bas de la bouche
        mouth_bottom = mouth_center + head_up * (0.15 * face_h)

        # =========================
        # YEUX (FRAME-BASED + STABLE)
        # =========================
        if right_eye is not None and left_eye is not None:
            points["eye_center"] = eye_center

            points["right_eye"] = right_eye
            points["left_eye"] = left_eye

            # inner/outer stable via head frame (pas vector brut)
            points["right_eye_inner"] = right_eye + head_right * (0.15 * face_w)
            points["right_eye_outer"] = right_eye - head_right * (0.25 * face_w)

            points["left_eye_inner"] = left_eye - head_right * (0.15 * face_w)
            points["left_eye_outer"] = left_eye + head_right * (0.25 * face_w)

        # =========================
        # EARS (IMPORTANT POUR HAI R MASK)
        # =========================
        if right_ear is not None and left_ear is not None:
            points["right_ear"] = right_ear
            points["left_ear"] = left_ear

        # =========================
        # NOSE + CHIN + HEAD CORE
        # =========================
        points["nose"] = nose

        points["hair_root"] = hair_root
        points["hair_left"] = hair_left
        points["hair_right"] = hair_right

        points["hair_top"] = hair_top
        points["left_top_hair3"] = left_top_hair3
        points["right_top_hair3"] = right_top_hair3

        points["head_center"] = head_center

        # chin approx (important pour hair mask stability)
        points["chin"] = nose + head_up * (1.6 * face_h)

        # =========================
        # MOUTH POINTS
        # =========================
        points.update({
            "mouth_center": mouth_center,
            "mouth_left_c": mouth_left,
            "mouth_left" : mouth_left,
            "mouth_right_c": mouth_right,
            "mouth_right" : mouth_right,
            "mouth_top": mouth_top,
            "mouth_bottom": mouth_bottom,
        })

        # =========================
        # TEMPORAL SMOOTHING
        # =========================
        prev = self.get_prev_facial_points()

        if prev is not None:
            for k in points:
                if k in prev:
                    points[k] = smooth * prev[k] + (1 - smooth) * points[k]

        # =========================
        # MEMORY UPDATE
        # =========================
        self.set_prev_facial_points(points)

        return points

    # ----------------- Calcul des points de la bouche -----------------
    def _calculate_mouth_width(self, right_ear, left_ear, right_eye, left_eye):
        """
        Calcule la largeur estimée de la bouche en fonction des points disponibles.
        """
        if right_ear is not None and left_ear is not None:
            return (left_ear[:, 0] - right_ear[:, 0]) * 0.4  # proportionnelle à la largeur de la tête
        elif right_eye is not None and left_eye is not None:
            return (left_eye[:, 0] - right_eye[:, 0]) * 0.5
        return torch.tensor(0.12, device=self.device).expand(self.B)  # fallback

    def _calculate_mouth_height(self, mouth_width):
        """
        Calcule la hauteur estimée de la bouche basée sur sa largeur.
        """
        return mouth_width * 0.25
    def estimate_missing_facial_points(self):
        """
        Estime les points manquants du visage (bouche, coins de lèvres, coins des yeux)
        à partir des points détectés (nez, yeux, oreilles, bouche si disponibles).

        Retourne un dictionnaire {nom_point: tensor [B,2]}.
        """
        estimated_points = {}
        B = self.B
        device = self.device

        # ----------------- Points de base -----------------
        nose = self.keypoints[:, 0, :2]  # point 0 = nez

        # BOUCHE : points détectés
        mouth_center = self.get_point(18)  # centre de la bouche (point 18)
        mouth_left = self.get_point(40)  # coin gauche de la bouche (point 40)
        mouth_right = self.get_point(41)  # coin droit de la bouche (point 41)

        mouth_left_ext = self.get_point(38)  # coin gauche de la bouche (point 38)
        mouth_right_ext = self.get_point(39)  # coin droit de la bouche (point 39)

        # YEUX : points détectés
        right_eye = self.get_point(14)  # œil droit (point 14)
        left_eye = self.get_point(15)   # œil gauche (point 15)

        # OREILLES : points détectés
        right_ear = self.get_point(16)  # oreille droite (point 16)
        left_ear = self.get_point(17)   # oreille gauche (point 17)

        # Lèvre du haut
        mouth_top_mid_r3 = self.get_point(70)  #
        mouth_top_mid_r2 = self.get_point(71)  #
        mouth_top_mid_r1 = self.get_point(72)  #
        mouth_top = self.get_point(73)  # 'mouth_top_mid' (point 73) comme centre de la lèvre supérieure
        mouth_top_mid_l1 = self.get_point(74)  #
        mouth_top_mid_l2 = self.get_point(75)  #
        mouth_top_mid_l3 = self.get_point(76)  #

        # Lèvre du bas
        mouth_bot_mid_r3 = self.get_point(77)  #
        mouth_bot_mid_r2 = self.get_point(78)  #
        mouth_bot_mid_r1 = self.get_point(79)  #
        mouth_bottom = self.get_point(80)  # 'mouth_bot_mid' (point 80) comme centre de la lèvre inférieure
        mouth_bot_mid_l1 = self.get_point(81)  #
        mouth_bot_mid_l2 = self.get_point(82)  #
        mouth_bot_mid_l3 = self.get_point(83)  #

        # ----------------- Largeur et Hauteur de la bouche -----------------
        mouth_width = self._calculate_mouth_width(right_ear, left_ear, right_eye, left_eye)
        mouth_height = self._calculate_mouth_height(mouth_width)

        # ----------------- LÈVRE DU HAUT ET DU BAS -----------------
        # Utilisation des points spécifiques 'mouth_top_mid' et 'mouth_bot_mid'

        if mouth_top is None or mouth_bottom is None:
            raise ValueError("Les points 'mouth_top_mid' ou 'mouth_bot_mid' sont manquants !")

        # Estimation des coins gauche/droit de la bouche
        mouth_left_corner = mouth_center.clone()
        mouth_left_corner[:, 0] -= mouth_width / 2

        mouth_right_corner = mouth_center.clone()
        mouth_right_corner[:, 0] += mouth_width / 2

        # Calcul des points intermédiaires pour les lèvres supérieures et inférieures
        estimated_points['mouth_center'] = mouth_center
        if mouth_left is not None and mouth_right is not None:
            estimated_points['mouth_left'] = mouth_left
            estimated_points['mouth_right'] = mouth_right

        # ----------------- Calcul des autres points de la lèvre supérieure -----------------
        if mouth_left_ext is not None and mouth_right_ext is not None:
            estimated_points['mouth_left_c'] = mouth_left_ext
            estimated_points['mouth_right_c'] = mouth_right_ext
        else:
            estimated_points['mouth_left_c'] = mouth_left_corner
            estimated_points['mouth_right_c'] = mouth_right_corner

        # ----------------- LÈVRE DU HAUT -----------------
        if mouth_top is not None:
            estimated_points['mouth_top_mid_r3'] = mouth_top_mid_r3  # point réel de la lèvre supérieure
            estimated_points['mouth_top_mid_r2'] = mouth_top_mid_r2  # point réel de la lèvre supérieure
            estimated_points['mouth_top_mid_r1'] = mouth_top_mid_r1  # point réel de la lèvre supérieure
            estimated_points['mouth_top'] = mouth_top  # point réel de la lèvre supérieure
            estimated_points['mouth_top_mid'] = mouth_top  # point réel de la lèvre supérieure
            estimated_points['mouth_top_mid_l1'] = mouth_top_mid_l1  # point réel de la lèvre supérieure
            estimated_points['mouth_top_mid_l2'] = mouth_top_mid_l2  # point réel de la lèvre supérieure
            estimated_points['mouth_top_mid_l3'] = mouth_top_mid_l3  # point réel de la lèvre supérieure
        else:
            # Ajustement des positions verticales pour la lèvre supérieure
            mouth_top_height = mouth_height * 0.25  # Ajustement basé sur la hauteur estimée de la bouche
            estimated_points['mouth_top_mid_r3'] = mouth_top.clone()
            estimated_points['mouth_top_mid_r2'] = mouth_top.clone()
            estimated_points['mouth_top_mid_r1'] = mouth_top.clone()
            estimated_points['mouth_top'] = mouth_top.clone()
            estimated_points['mouth_top_mid'] = mouth_top.clone()
            estimated_points['mouth_top_mid_l1'] = mouth_top.clone()
            estimated_points['mouth_top_mid_l2'] = mouth_top.clone()
            estimated_points['mouth_top_mid_l3'] = mouth_top.clone()

            estimated_points['mouth_top_mid_r3'][:, 1] -= mouth_top_height
            estimated_points['mouth_top_mid_r2'][:, 1] -= mouth_top_height * 0.75
            estimated_points['mouth_top_mid_r1'][:, 1] -= mouth_top_height * 0.5
            estimated_points['mouth_top_mid'][:, 1] -= mouth_top_height * 0.25
            estimated_points['mouth_top_mid_l1'][:, 1] += mouth_top_height * 0.25
            estimated_points['mouth_top_mid_l2'][:, 1] += mouth_top_height * 0.5
            estimated_points['mouth_top_mid_l3'][:, 1] += mouth_top_height * 0.75

        # Lèvre du bas
        if mouth_bottom is not None:
            estimated_points['mouth_bot_mid_r3'] = mouth_bot_mid_r3  # point réel de la lèvre inférieure
            estimated_points['mouth_bot_mid_r2'] = mouth_bot_mid_r2  # point réel de la lèvre inférieure
            estimated_points['mouth_bot_mid_r1'] = mouth_bot_mid_r1  # point réel de la lèvre inférieure
            estimated_points['mouth_bottom'] = mouth_bottom  # point réel de la lèvre inférieure
            estimated_points['mouth_bot_mid'] = mouth_bottom  # point réel de la lèvre inférieure
            estimated_points['mouth_bot_mid_l1'] = mouth_bot_mid_l1  # point réel de la lèvre inférieure
            estimated_points['mouth_bot_mid_l2'] = mouth_bot_mid_l2  # point réel de la lèvre inférieure
            estimated_points['mouth_bot_mid_l3'] = mouth_bot_mid_l3  # point réel de la lèvre inférieure
        else:
            # ----------------- LÈVRE DU BAS -----------------
            # Ajustement des positions verticales pour la lèvre inférieure
            mouth_bottom_height = mouth_height * 0.25  # Ajustement basé sur la hauteur estimée de la bouche
            estimated_points['mouth_bot_mid_r3'] = mouth_bottom.clone()
            estimated_points['mouth_bot_mid_r2'] = mouth_bottom.clone()
            estimated_points['mouth_bot_mid_r1'] = mouth_bottom.clone()
            estimated_points['mouth_bottom'] = mouth_bottom.clone()
            estimated_points['mouth_bot_mid'] = mouth_bottom.clone()
            estimated_points['mouth_bot_mid_l1'] = mouth_bottom.clone()
            estimated_points['mouth_bot_mid_l2'] = mouth_bottom.clone()
            estimated_points['mouth_bot_mid_l3'] = mouth_bottom.clone()

            estimated_points['mouth_bot_mid_r3'][:, 1] -= mouth_bottom_height
            estimated_points['mouth_bot_mid_r2'][:, 1] -= mouth_bottom_height * 0.75
            estimated_points['mouth_bot_mid_r1'][:, 1] -= mouth_bottom_height * 0.5
            estimated_points['mouth_bot_mid'][:, 1] -= mouth_bottom_height * 0.25
            estimated_points['mouth_bot_mid_l1'][:, 1] += mouth_bottom_height * 0.25
            estimated_points['mouth_bot_mid_l2'][:, 1] += mouth_bottom_height * 0.5
            estimated_points['mouth_bot_mid_l3'][:, 1] += mouth_bottom_height * 0.75

        return estimated_points

    # ----------------- Keypoint utils -----------------
    def get_point(self, idx):
        if isinstance(idx, str):
            idx = self.FACIAL_POINT_IDX[idx]
        return self.keypoints[:, idx, :2]

    # ----------------- Torso angle -----------------
    def compute_torso_angle(self):
        """Calcule l'angle du torse selon les épaules"""
        r_shoulder = self.get_point(2)
        l_shoulder = self.get_point(5)
        vec = r_shoulder - l_shoulder
        angle = torch.atan2(vec[:,1], vec[:,0]).unsqueeze(1)  # [B,1]
        self.angles = angle
        return angle
    # ----------------- Masque Bouche -----------------



    def get_mouth_region(
            self,
            H: int,
            W: int,
            device=None,
            debug: bool = False,
            verbose: bool = False,
            debug_dir: str = None,
            frame_counter: int = 0,
            min_size=8,
            min_area=120,
            temporal_smooth=0.8,
            profile_strength=0.08,
            scale=8
        ):

        # Valeurs par défaut dynamiques
        expand_w = self.get_bouche_expand_w()  # 0.60
        expand_h = self.get_bouche_expand_h()  # 0.30

        if device is None:
            device = self.device

        B = self.B
        mask = torch.zeros((B, 1, H, W), device=device)

        try:
            points_dict = self.estimate_missing_facial_points()
        except Exception as e:
            print(f"[WARN] mouth_region: estimation failed → fallback empty ({e})")
            return mask

        required_keys = [
            'mouth_left', 'mouth_right', 'mouth_center',
            'mouth_top_mid_r3', 'mouth_top_mid_r2', 'mouth_top_mid_r1',
            'mouth_top_mid', 'mouth_top_mid_l1', 'mouth_top_mid_l2', 'mouth_top_mid_l3',
            'mouth_bot_mid_r3', 'mouth_bot_mid_r2', 'mouth_bot_mid_r1', 'mouth_bot_mid',
            'mouth_bot_mid_l1', 'mouth_bot_mid_l2', 'mouth_bot_mid_l3',
        ]

        if not all(k in points_dict for k in required_keys):
            print(f"[ERROR] Missing required points in points_dict. Available keys: {list(points_dict.keys())}")
            return mask

        # Extraction des points de lèvres
        mouth_center = points_dict['mouth_center']
        mouth_left = points_dict['mouth_left']
        mouth_right = points_dict['mouth_right']

        # Lèvres du haut
        mouth_top_mid_r3 = points_dict['mouth_top_mid_r3']
        mouth_top_mid_r2 = points_dict['mouth_top_mid_r2']
        mouth_top_mid_r1 = points_dict['mouth_top_mid_r1']
        mouth_top_mid = points_dict['mouth_top_mid']
        mouth_top_mid_l1 = points_dict['mouth_top_mid_l1']
        mouth_top_mid_l2 = points_dict['mouth_top_mid_l2']
        mouth_top_mid_l3 = points_dict['mouth_top_mid_l3']

        # Lèvres du bas
        mouth_bot_mid_r3 = points_dict['mouth_bot_mid_r3']
        mouth_bot_mid_r2 = points_dict['mouth_bot_mid_r2']
        mouth_bot_mid_r1 = points_dict['mouth_bot_mid_r1']
        mouth_bot_mid = points_dict['mouth_bot_mid']
        mouth_bot_mid_l1 = points_dict['mouth_bot_mid_l1']
        mouth_bot_mid_l2 = points_dict['mouth_bot_mid_l2']
        mouth_bot_mid_l3 = points_dict['mouth_bot_mid_l3']

        # -----------------------------------------------------
        # Pour la partie inférieure de la bouche (bas)
        # -----------------------------------------------------
        lower_points = [
            mouth_right, mouth_bot_mid_r3, mouth_bot_mid_r2, mouth_bot_mid_r1, mouth_bot_mid,
            mouth_bot_mid_l1, mouth_bot_mid_l2, mouth_bot_mid_l3, mouth_left
        ]

        # -----------------------------------------------------
        # Pour la partie supérieure de la bouche (haut)
        # -----------------------------------------------------
        upper_points = [
            mouth_right, mouth_top_mid_r3, mouth_top_mid_r2, mouth_top_mid_r1, mouth_top_mid,
            mouth_top_mid_l1, mouth_top_mid_l2, mouth_top_mid_l3, mouth_left
        ]

        # Combine les points du bas et du haut
        all_points = lower_points + upper_points

        # Si des points sont NaN, ignorer l'image
        pts = torch.stack(all_points)
        if torch.isnan(pts).any():
            print(f"[ERROR] NaN values detected in the points. Skipping this batch.")
            return mask

        # Initialisation de l'état de la bouche
        if not hasattr(self, "_mouth_state"):
            self._mouth_state = {}

        for b in range(B):
            # Convertir les points en format numpy pour OpenCV
            # Mise à l'échelle correcte : chaque coordonnée est multipliée par H ou W
            points_np = np.array([
                [int(p[b, 0] * W), int(p[b, 1] * H)] for p in all_points
            ], dtype=np.int32)

            # Créer un masque vide
            img_mask = np.zeros((H, W), dtype=np.uint8)

            # Dessiner le polygone sur le masque (remplissage)
            cv2.fillPoly(img_mask, [points_np], color=1)

            # Convertir en tensor et ajouter à notre masque final
            mask[b, 0] = torch.tensor(img_mask, device=device, dtype=torch.float32)

            # -----------------------------------------------------
            # Débogage des points et du masque
            if verbose:
                print(f"[DEBUG][MOUTH][FRAME {frame_counter}] Points de la bouche pour la frame {b}:")
                for name, point in zip(
                    ["mouth_right", "mouth_bot_mid_r3", "mouth_bot_mid_r2", "mouth_bot_mid_r1", "mouth_bot_mid",
                    "mouth_bot_mid_l1", "mouth_bot_mid_l2", "mouth_bot_mid_l3", "mouth_left", "mouth_top_mid_r3",
                    "mouth_top_mid_r2", "mouth_top_mid_r1", "mouth_top_mid", "mouth_top_mid_l1", "mouth_top_mid_l2",
                    "mouth_top_mid_l3"], all_points):
                    print(f"{name}: {point[b].tolist()}")

                print(f"[DEBUG][MOUTH][FRAME {frame_counter}] Mask preview:")
                print(img_mask)

            # -----------------------------------------------------
            # Sauvegarder l'état de la bouche pour lissage temporel
            self._mouth_state[b] = {
                "center": mouth_center[b].detach(),
                "width": (mouth_right[b, 0] - mouth_left[b, 0]).detach(),
                "height": (mouth_bot_mid[b, 1] - mouth_top_mid[b, 1]).detach()
            }

        mask = mask.clamp(0, 1)

        # Sauvegarder le masque en mode debug
        if debug and debug_dir is not None and frame_counter == 0:
            save_debug_mask_scale( mask=mask, debug_dir=debug_dir, frame_counter=frame_counter, name="mouth_region_mask", scale=scale )

        return mask




    def compute_mouth_delta(
        pose,
        mask_mouth,
        H,
        W,
        frame_counter,
        device,
        smooth=0.85,
        strength=2.0,
        debug=False,
        mouth_state=None,   # 🔥 NEW (optionnel mais crucial)
    ):
        """
        Compute mouth motion field (NO WARP).
        Safe for accumulation → avoids blur.
        """

        # =========================
        # 1. Facial points
        # =========================
        facial_points = pose.estimate_facial_points_full(smooth=smooth)

        mouth_left  = facial_points['mouth_left_c']
        mouth_right = facial_points['mouth_right_c']

        # =========================
        # 🔥 ALIGNEMENT AVEC get_mouth_region (profil correction)
        # =========================
        if mouth_state is not None and 'center' in mouth_state:

            if debug:
                print("[DEBUG][MOUTH] 🔥 using mouth_state alignment (profile-aware)")

            mouth_center = mouth_state['center']  # [B,2]

            # direction bouche (axe horizontal stable)
            direction = torch.nn.functional.normalize(
                mouth_right - mouth_left, dim=-1
            )

            width = torch.norm(mouth_right - mouth_left, dim=-1, keepdim=True)

            mouth_left  = mouth_center - direction * (width * 0.5)
            mouth_right = mouth_center + direction * (width * 0.5)

        # =========================
        # 2. MASK (stable + clean)
        # =========================
        mask = feather_inside_strict2(mask_mouth, radius=3, blur_kernel=5, sigma=1.2)
        mask = feather_outside_only_alpha2(mask, radius=2, sigma=1.0)

        mask_mean = mask.mean()

        # auto boost léger mais contrôlé
        boost = torch.clamp(0.04 / (mask_mean + 1e-6), 1.0, 3.0)
        mask = torch.clamp(mask * boost, 0, 1)

        mask_exp = mask.permute(0, 2, 3, 1)

        # =========================
        # 3. TIME SIGNALS (smooth)
        # =========================
        t = frame_counter / 10.0

        t_smile  = torch.sin(torch.tensor(t * 1.2, device=device))
        t_breath = torch.sin(torch.tensor(t * 0.6, device=device))
        t_open   = torch.sin(torch.tensor(t * 0.4, device=device))

        # =========================
        # 4. COORD GRID
        # =========================
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        x = x[None]
        y = y[None]

        mouth_left_px  = mouth_left  * torch.tensor([W-1, H-1], device=device)
        mouth_right_px = mouth_right * torch.tensor([W-1, H-1], device=device)

        # =========================
        # 5. DISTANCE FIELD
        # =========================
        sigma = 16.0

        def gaussian(cx, cy):
            return torch.exp(-(
                (x - cx[:, None, None])**2 +
                (y - cy[:, None, None])**2
            ) / (2 * sigma**2))

        dist_left  = gaussian(mouth_left_px[:,0],  mouth_left_px[:,1])
        dist_right = gaussian(mouth_right_px[:,0], mouth_right_px[:,1])

        # =========================
        # 6. MOTION FIELD
        # =========================
        amp = 0.25 * strength

        delta = torch.zeros((mask.shape[0], H, W, 2), device=device)

        # smile horizontal
        delta[..., 0] += amp * t_smile * (dist_right - dist_left)

        # smile vertical léger
        delta[..., 1] += 0.12 * amp * t_smile * (dist_left + dist_right)

        # ouverture + respiration
        delta[..., 1] += (
            0.025 * strength * t_breath +
            0.02  * strength * torch.abs(t_open)
        ) * mask_exp[..., 0]

        # =========================
        # 7. APPLY MASK
        # =========================
        delta = delta * mask_exp

        # =========================
        # 8. NORMALISATION SAFE (ANTI BLUR)
        # =========================
        delta = torch.tanh(delta) * 2.0
        delta = torch.clamp(delta, -3.0, 3.0)

        # =========================
        # DEBUG LOGS (inchangés + enrichis léger)
        # =========================
        if debug:
            print("[DEBUG][MOUTH DELTA CLEAN]")
            print(f"  mean={delta.abs().mean().item():.6f}")
            print(f"  max={delta.abs().max().item():.6f}")
            print(f"  mask_mean={mask_mean.item():.5f}")

            if mouth_state is not None:
                print(f"  profile_sync=ON")
            else:
                print(f"  profile_sync=OFF")

        return delta, facial_points

    def compute_mouth_delta(
        pose,
        mask_mouth,
        H,
        W,
        frame_counter,
        device,
        smooth=0.85,
        strength=2.0,
        debug=False
    ):
        """
        Compute mouth motion field (NO WARP).
        Safe for accumulation → avoids blur.
        """

        # =========================
        # 1. Facial points
        # =========================
        facial_points = pose.estimate_facial_points_full(smooth=smooth)

        mouth_left  = facial_points['mouth_left_c']
        mouth_right = facial_points['mouth_right_c']

        # =========================
        # 2. MASK (stable + clean)
        # =========================
        mask = feather_inside_strict2(mask_mouth, radius=3, blur_kernel=5, sigma=1.2)
        mask = feather_outside_only_alpha2(mask, radius=2, sigma=1.0)

        mask_mean = mask.mean()

        # auto boost léger mais contrôlé
        boost = torch.clamp(0.04 / (mask_mean + 1e-6), 1.0, 3.0)
        mask = torch.clamp(mask * boost, 0, 1)

        mask_exp = mask.permute(0, 2, 3, 1)

        # =========================
        # 3. TIME SIGNALS (smooth)
        # =========================
        t = frame_counter / 10.0

        t_smile  = torch.sin(torch.tensor(t * 1.2, device=device))
        t_breath = torch.sin(torch.tensor(t * 0.6, device=device))
        t_open   = torch.sin(torch.tensor(t * 0.4, device=device))

        # =========================
        # 4. COORD GRID
        # =========================
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        x = x[None]
        y = y[None]

        mouth_left_px  = mouth_left  * torch.tensor([W-1, H-1], device=device)
        mouth_right_px = mouth_right * torch.tensor([W-1, H-1], device=device)

        # =========================
        # 5. DISTANCE FIELD
        # =========================
        sigma = 16.0

        def gaussian(cx, cy):
            return torch.exp(-(
                (x - cx[:, None, None])**2 +
                (y - cy[:, None, None])**2
            ) / (2 * sigma**2))

        dist_left  = gaussian(mouth_left_px[:,0],  mouth_left_px[:,1])
        dist_right = gaussian(mouth_right_px[:,0], mouth_right_px[:,1])

        # =========================
        # 6. MOTION FIELD
        # =========================
        amp = 0.25 * strength

        delta = torch.zeros((mask.shape[0], H, W, 2), device=device)

        # smile horizontal
        delta[..., 0] += amp * t_smile * (dist_right - dist_left)

        # smile vertical léger
        delta[..., 1] += 0.12 * amp * t_smile * (dist_left + dist_right)

        # ouverture + respiration
        delta[..., 1] += (
            0.025 * strength * t_breath +
            0.02  * strength * torch.abs(t_open)
        ) * mask_exp[..., 0]

        # =========================
        # 7. APPLY MASK
        # =========================
        delta = delta * mask_exp

        # =========================
        # 8. NORMALISATION SAFE (ANTI BLUR)
        # =========================
        delta = torch.tanh(delta) * 2.0

        # clamp final (important)
        delta = torch.clamp(delta, -3.0, 3.0)

        if debug:
            print("[DEBUG][MOUTH DELTA CLEAN]")
            print(f"  mean={delta.abs().mean().item():.6f}")
            print(f"  max={delta.abs().max().item():.6f}")
            print(f"  mask_mean={mask_mean.item():.5f}")

        return delta, facial_points

    # ----------------- Torso delta -----------------
    # Attention le expand_w et le shrink_h doit être similaire pour compute_torso_delta et create_upper_body_mask
    def compute_torso_delta(
        self,
        latent_h: int,
        latent_w: int,
        expand_w=None,
        shrink_h=None
    ):
        """
        Delta torse aligné avec le masque ovale adaptatif
        - largeur adaptative épaules/hanche
        - centre pondéré chest/belly
        - lissage temporel possible
        """
        # Valeurs par défaut dynamiques
        if expand_w is None:
            expand_w = self.get_torce_expand_w()
        if shrink_h is None:
            shrink_h = self.get_torce_shrink_h()

        # =========================
        # 🔹 Points torse
        # =========================
        r_sh = self.get_point(19)  # right_shoulder
        l_sh = self.get_point(20)  # left_shoulder
        r_hip = self.get_point(8)  # right_hip
        l_hip = self.get_point(11) # left_hip

        # Stack pour traitement batch
        pts = torch.stack([r_sh, l_sh, r_hip, l_hip], dim=1)  # [B,4,2]

        # =========================
        # 🔹 Centres anatomiques
        # =========================
        shoulder_center = (r_sh + l_sh) / 2
        hip_center = (r_hip + l_hip) / 2

        chest = shoulder_center * 0.7 + hip_center * 0.3
        belly = shoulder_center * 0.3 + hip_center * 0.7

        # Centre pondéré
        center = (chest + belly) / 2  # [B,2]

        # =========================
        # 🔹 Largeur/hauteur adaptative
        # =========================
        shoulder_width = torch.norm(r_sh - l_sh, dim=1, keepdim=True)  # [B,1]
        hip_width = torch.norm(r_hip - l_hip, dim=1, keepdim=True)    # [B,1]

        width_top = shoulder_width * expand_w
        width_bottom = hip_width * expand_w * 0.9
        height = torch.norm(hip_center - shoulder_center, dim=1, keepdim=True) * shrink_h

        # =========================
        # 🔹 Bounding box du torse (adapté à l’ellipse)
        # =========================
        x_min = center[:,0:1] - width_top/2
        x_max = center[:,0:1] + width_top/2
        y_min = center[:,1:2] - height/2
        y_max = center[:,1:2] + height/2

        # Recalcule centre exact
        torso_center_x = (x_min + x_max) * 0.5
        torso_center_y = (y_min + y_max) * 0.5
        torso_center = torch.cat([torso_center_x, torso_center_y], dim=1)  # [B,2]

        # =========================
        # 🔹 Delta normalisé
        # =========================
        delta = torso_center - 0.5
        delta = torch.tanh(delta * 1.5) * 0.12

        # =========================
        # 🔹 Smooth temporel (optionnel)
        # =========================
        if hasattr(self, "_prev_torso_delta"):
            alpha = 0.85
            delta = alpha * self._prev_torso_delta + (1 - alpha) * delta

        self._prev_torso_delta = delta
        self.delta = delta

        return delta


    # --------------- Mask decor ---------------------------------------------
    def create_decor_mask( self, H: int, W: int, mask_face=None, mask_torso=None, mask_hair=None, debug: bool = False, debug_dir: str = None, frame_counter: int = 0,
                          expand=4.2, vertical_bias=2.0, falloff_strength=3.0 ):
        device = self.device
        B = self.B

        mask = torch.zeros(B, 1, H, W, device=device)

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )

        def to_px(kp):
            return kp * torch.tensor([W - 1, H - 1], device=device)

        for b in range(B):

            # 🔹 Points clés élargis
            body_points = [
                "nose", "chin", "neck", "left_side_neck", "right_side_neck", "anchor_neck",
                "right_shoulder", "left_shoulder",
                "right_elbow", "left_elbow",
                "right_wrist", "left_wrist",
                "right_hip", "left_hip",
                "right_knee", "left_knee",
                "right_ankle", "left_ankle"
            ]

            pts = torch.stack([to_px(self.get_point(self.FACIAL_POINT_IDX[p])[b]) for p in body_points])
            # 🔹 Bounding
            min_xy = pts.min(dim=0).values
            max_xy = pts.max(dim=0).values
            center = (min_xy + max_xy) / 2

            width = (max_xy[0] - min_xy[0]) * expand
            height = (max_xy[1] - min_xy[1]) * expand * vertical_bias
            # 🔹 Ellipse corps
            dx = xx - center[0]
            dy = yy - center[1]

            ellipse = (dx / (width / 2 + 1e-6))**2 + (dy / (height / 2 + 1e-6))**2
            inside = torch.exp(-ellipse * falloff_strength)
            # 🔥 Décor = extérieur
            mask[b, 0] = 1.0 - inside

        # =========================
        # 🔹 Exclusion simple
        # =========================
        if mask_face is not None:
            mask *= (1.0 - mask_face)
        if mask_torso is not None:
            mask *= (1.0 - mask_torso)
        if mask_hair is not None:
            mask *= (1.0 - mask_hair)

        # =========================
        # 🔹 Edge falloff (clé)
        # =========================
        yy_n, xx_n = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )

        radial = torch.sqrt(xx_n**2 + yy_n**2)
        edge_falloff = (1.0 - radial).clamp(0, 1)

        mask *= edge_falloff.unsqueeze(0).unsqueeze(0)

        # =========================
        # 🔹 Smooth léger
        # =========================
        mask = torch.nn.functional.avg_pool2d(
            mask,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # =========================
        # 🔹 Clamp final
        # =========================
        mask = torch.clamp(mask, 0.0, 1.0)

        # =========================
        # 🔹 Debug
        # =========================
        if debug and debug_dir is not None and frame_counter == 0:
            save_debug_mask(mask, H, W, debug_dir, frame_counter, prefix="decor_mask")

        return mask
    # -------- version pose -----------------------------------
    def create_decor_outpose_mask( self, H: int, W: int, debug: bool = False, debug_dir: str = None, frame_counter: int = 0, expand=2.2, vertical_bias=1.2, falloff_strength=2.0 ):
        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        yy, xx = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij"
        )

        def to_px(kp):
            return kp * torch.tensor([W-1, H-1], device=self.device)

        for b in range(self.B):

            # =========================
            # 🔹 Points principaux
            # =========================
            r_sh = to_px(self.get_point(self.FACIAL_POINT_IDX["right_shoulder"])[b])
            l_sh = to_px(self.get_point(self.FACIAL_POINT_IDX["left_shoulder"])[b])
            r_hip = to_px(self.get_point(self.FACIAL_POINT_IDX["right_hip"])[b])
            l_hip = to_px(self.get_point(self.FACIAL_POINT_IDX["left_hip"])[b])

            neck = to_px(self.get_point(self.FACIAL_POINT_IDX["neck"])[b])
            head = to_px(self.get_point(self.FACIAL_POINT_IDX["nose"])[b])

            # =========================
            # 🔹 Bras (optionnel mais crucial)
            # =========================
            points_width = [r_sh, l_sh, r_hip, l_hip]

            if self.has_point("right_elbow"):
                points_width.append(to_px(self.get_point(self.FACIAL_POINT_IDX["right_elbow"])[b]))
            if self.has_point("left_elbow"):
                points_width.append(to_px(self.get_point(self.FACIAL_POINT_IDX["left_elbow"])[b]))

            pts = torch.stack(points_width)

            # =========================
            # 🔹 Bounding dynamique
            # =========================
            min_xy = pts.min(dim=0).values
            max_xy = pts.max(dim=0).values

            center = (min_xy + max_xy) / 2

            width = (max_xy[0] - min_xy[0]) * expand
            height = (max_xy[1] - head[1]) * expand * vertical_bias

            # =========================
            # 🔹 Ellipse
            # =========================
            dx = xx - center[0]
            dy = yy - center[1]

            ellipse = (dx / (width / 2 + 1e-6))**2 + (dy / (height / 2 + 1e-6))**2

            inside = torch.exp(-ellipse * falloff_strength)

            # 🔥 inversion
            outside = 1.0 - inside
            outside = torch.clamp(outside, 0, 1)

            mask[b, 0] = outside

        # =========================
        # 🔹 Feather
        # =========================
        mask = feather_inside_strict(mask, radius=8, blur_kernel=5, sigma=2.0)

        if debug and debug_dir is not None and frame_counter == 0:
            save_debug_mask(mask, H, W, debug_dir, frame_counter, prefix="decor_mask")

        return mask

    def create_upper_body_mask(
        self,
        H: int,
        W: int,
        debug: bool = False,
        debug_dir: str = None,
        frame_counter: int = 0,
        expand_w=None,
        shrink_h=None,
        roundness=0.8
    ):
        """
        Masque torse PRO++ :
        - largeur adaptative épaules / hanches
        - forme elliptique anatomique
        - pondération gaussienne
        """
        # Valeurs par défaut dynamiques
        expand_w = expand_w or self.get_torce_expand_w()
        shrink_h = shrink_h or self.get_torce_shrink_h()

        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        # Grille unique
        yy, xx = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij"
        )

        def to_px(kp):
            return kp * torch.tensor([W-1, H-1], device=self.device)

        for b in range(self.B):
            r_sh = to_px(self.get_point(19)[b])
            l_sh = to_px(self.get_point(20)[b])
            r_hip = to_px(self.get_point(8)[b])
            l_hip = to_px(self.get_point(11)[b])

            # Centres anatomiques et largeur/hauteur
            shoulder_center, hip_center = (r_sh + l_sh)/2, (r_hip + l_hip)/2
            #center = (shoulder_center * 0.5 + hip_center * 0.5)
            # Centre anatomique corrigé
            center = shoulder_center * 0.65 + hip_center * 0.35 # new code

            width_top = torch.norm(r_sh - l_sh) * expand_w
            width_bottom = torch.norm(r_hip - l_hip) * expand_w * 0.9
            #height = torch.norm(hip_center - shoulder_center) * shrink_h

            # Hauteur sécurisée # new code
            base_height = torch.norm(hip_center - shoulder_center)
            height = torch.clamp(base_height * shrink_h, min=H * 0.25)

            # Légère remontée du torse
            center[1] -= height * 0.08

            # Ellipse adaptative
            dx, dy = xx - center[0], yy - center[1]
            t = torch.clamp((dy / height) + 0.5, 0, 1)
            width_interp = width_top * (1 - t) + width_bottom * t
            ellipse = (dx / (width_interp / 2))**2 + (dy / (height / 2))**2

            #mask[b,0] = torch.exp(-ellipse * 2.5)  # Gaussian falloff
            mask[b,0] = torch.exp(-ellipse * 1.5) # new code

        # Clamp + feather
        mask = torch.clamp(mask, 0, 1)
        mask = feather_inside_strict(mask, radius=5, blur_kernel=3, sigma=1.2)

        # Debug
        if debug and debug_dir is not None and frame_counter == 0:
            save_debug_mask(mask, H, W, debug_dir, frame_counter, prefix="torso_mask_PRO")


        return mask


    def create_hair_mask(
        self,
        H: int,
        W: int,
        debug: bool = False,
        debug_dir: str = None,
        frame_counter: int = 0,
        top_extend=0.55,
        side_extend=0.25,
        height_factor=1.15,
    ):
        """
        Hair mask basé sur keypoints :
        - ears + eyes + nose → orientation de tête
        - ellipse inclinée (tilt head-aware)
        - centre stabilisé

            'hair_top': 28, # millieu centre

            'left_top_hair1': 31, # premier point
            'left_top_hair2': 32, # Point gauche au sommet, 2ème point centre gauche
            'left_top_hair3': 33, # point le plus eloigné

            'right_top_hair1': 34, # premier point
            'right_top_hair2': 35, # Point gauche au sommet, 2ème point centre droit
            'right_top_hair3': 36, # point le plus eloigné
        """

        mask_face = self.create_face_mask(
            H, W,
            debug=debug,
            debug_dir=debug_dir,
            frame_counter=frame_counter
        )

        mask_hair = torch.zeros_like(mask_face)

        for b in range(self.B):

            fp = self.estimate_facial_points_full(smooth=True)

            # =========================
            # KEYPOINTS SAFE (FIX IMPORTANT)
            # =========================
             # Calcul des points pour les cheveux
            hair_top = ensure_2d(fp['hair_top'])
            hair_left = ensure_2d(fp['left_top_hair3'])
            hair_right = ensure_2d(fp['right_top_hair3'])
            hair_root = ensure_2d(fp['hair_root'])
            nose = ensure_2d(fp['nose'])
            left_eye = ensure_2d(fp['left_eye'])
            right_eye = ensure_2d(fp['right_eye'])
            left_ear = ensure_2d(fp['left_ear'])
            right_ear = ensure_2d(fp['right_ear'])

            # skip si invalid
            if any(k is None for k in [hair_top, hair_left, hair_right, hair_root, nose, left_eye, right_eye, left_ear, right_ear]):
                continue

            # =========================
            # pixels conversion
            # =========================
            def to_px(p):
                return p * torch.tensor([W - 1, H - 1], device=self.device)

            lhair_px = to_px(hair_left)
            rhair_px = to_px(hair_right)
            hair_px = to_px(hair_root)
            nose_px = to_px(nose)
            leye_px = to_px(left_eye)
            reye_px = to_px(right_eye)
            lear_px = to_px(left_ear)
            rear_px = to_px(right_ear)

            # =========================
            # HEAD CENTER ROBUST
            # =========================
            hair_center = (lhair_px + rhair_px) * 0.5
            eye_center = (leye_px + reye_px) * 0.5
            ear_center = (lear_px + rear_px) * 0.5
            head_center = (nose_px + eye_center + ear_center) / 3.0

            cx = head_center[0]
            cy = head_center[1]

            # =========================
            # HEAD ORIENTATION
            # =========================
            eye_vec = reye_px - leye_px
            angle = torch.atan2(eye_vec[1], eye_vec[0])
            angle_deg = angle * 180.0 / 3.14159265

            # =========================
            # HEAD SIZE
            # =========================
            head_width = torch.norm(rear_px - lear_px)
            head_height = torch.norm(nose_px - hair_center) #eye_center

            h_face = head_height * 2.0
            w_face = head_width * 1.6

            # =========================
            # HAIR EXTENSION
            # =========================
            rx = (w_face * 0.5) * (1.0 + side_extend)
            ry = (h_face * 0.5) * (1.0 + top_extend) * height_factor

            cy = cy - h_face * 0.55  # shift scalp upward

            # =========================
            # SAFE CLAMP
            # =========================
            cx = int(torch.clamp(cx, 0, W - 1).item())
            cy = int(torch.clamp(cy, 0, H - 1).item())
            rx = max(2, int(rx.item()))
            ry = max(2, int(ry.item()))

            # =========================
            # ELLIPSE MASK
            # =========================
            mask_np = np.zeros((H, W), dtype=np.uint8)

            cv2.ellipse(
                mask_np,
                (cx, cy),
                (rx, ry),
                angle=float(angle_deg),
                startAngle=0,
                endAngle=360,
                color=255,
                thickness=-1
            )

            mask_hair[b, 0] = torch.from_numpy(mask_np / 255.0).to(self.device)

        # =========================
        # REMOVE FACE AREA
        # =========================
        mask_hair = mask_hair * (1.0 - mask_face)

        # =========================
        # SOFTENING
        # =========================
        mask_hair = mask_hair ** 2.1
        mask_hair = feather_outside_only_alpha(mask_hair, radius=6, sigma=2.2)

        # =========================
        # DEBUG
        # =========================
        if debug and debug_dir is not None and frame_counter == 0:
            save_debug_mask( mask_hair, H, W, debug_dir, frame_counter, prefix="hair_mask_keypoint_" )

        return mask_hair

    def create_hair_mask_ori(
        self,
        H: int,
        W: int,
        debug: bool = False,
        debug_dir: str = None,
        frame_counter: int = 0,
        top_extend=0.55,
        side_extend=0.25,
        height_factor=1.15,
    ):
        """
        Hair mask basé sur keypoints :
        - ears + eyes + nose → orientation de tête
        - ellipse inclinée (tilt head-aware)
        - centre stabilisé
        """

        mask_face = self.create_face_mask(
            H, W,
            debug=debug,
            debug_dir=debug_dir,
            frame_counter=frame_counter
        )

        mask_hair = torch.zeros_like(mask_face)

        for b in range(self.B):

            fp = self.estimate_facial_points_full(smooth=True)

            # =========================
            # KEYPOINTS SAFE (FIX IMPORTANT)
            # =========================
            nose = ensure_2d(fp['nose'])
            left_eye = ensure_2d(fp['left_eye'])
            right_eye = ensure_2d(fp['right_eye'])
            left_ear = ensure_2d(fp['left_ear'])
            right_ear = ensure_2d(fp['right_ear'])

            # skip si invalid
            if any(k is None for k in [nose, left_eye, right_eye, left_ear, right_ear]):
                continue

            # =========================
            # pixels conversion
            # =========================
            def to_px(p):
                return p * torch.tensor([W - 1, H - 1], device=self.device)

            nose_px = to_px(nose)
            leye_px = to_px(left_eye)
            reye_px = to_px(right_eye)
            lear_px = to_px(left_ear)
            rear_px = to_px(right_ear)

            # =========================
            # HEAD CENTER ROBUST
            # =========================
            eye_center = (leye_px + reye_px) * 0.5
            ear_center = (lear_px + rear_px) * 0.5
            head_center = (nose_px + eye_center + ear_center) / 3.0

            cx = head_center[0]
            cy = head_center[1]

            # =========================
            # HEAD ORIENTATION
            # =========================
            eye_vec = reye_px - leye_px
            angle = torch.atan2(eye_vec[1], eye_vec[0])
            angle_deg = angle * 180.0 / 3.14159265

            # =========================
            # HEAD SIZE
            # =========================
            head_width = torch.norm(reye_px - leye_px)
            head_height = torch.norm(nose_px - eye_center)

            h_face = head_height * 2.2
            w_face = head_width * 1.6

            # =========================
            # HAIR EXTENSION
            # =========================
            rx = (w_face * 0.5) * (1.0 + side_extend)
            ry = (h_face * 0.5) * (1.0 + top_extend) * height_factor

            cy = cy - h_face * 0.55  # shift scalp upward

            # =========================
            # SAFE CLAMP
            # =========================
            cx = int(torch.clamp(cx, 0, W - 1).item())
            cy = int(torch.clamp(cy, 0, H - 1).item())
            rx = max(2, int(rx.item()))
            ry = max(2, int(ry.item()))

            # =========================
            # ELLIPSE MASK
            # =========================
            mask_np = np.zeros((H, W), dtype=np.uint8)

            cv2.ellipse(
                mask_np,
                (cx, cy),
                (rx, ry),
                angle=float(angle_deg),
                startAngle=0,
                endAngle=360,
                color=255,
                thickness=-1
            )

            mask_hair[b, 0] = torch.from_numpy(mask_np / 255.0).to(self.device)

        # =========================
        # REMOVE FACE AREA
        # =========================
        mask_hair = mask_hair * (1.0 - mask_face)

        # =========================
        # SOFTENING
        # =========================
        mask_hair = mask_hair ** 2.1
        mask_hair = feather_outside_only_alpha(mask_hair, radius=6, sigma=2.2)

        # =========================
        # DEBUG
        # =========================
        if debug and debug_dir is not None and frame_counter == 0:
            save_debug_mask( mask_hair, H, W, debug_dir, frame_counter, prefix="hair_mask_keypoint_" )

        return mask_hair

    # version dynamique pro
    def create_left_eye_mask(self, H: int, W: int, debug=False, debug_dir=None, frame_counter=0, expand_w=0.3, expand_h=0.3):
        """
        Masque dynamique pour l’œil gauche, ovale réaliste.
        """
        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        for b in range(self.B):
            points = [self.get_point(15)[b].cpu().numpy()]  # left_eye keypoints
            pts = np.array([[p[0]*(W-1), p[1]*(H-1)] for p in points], dtype=np.float32)

            if len(pts) == 0:
                continue

            cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
            w_eye = np.max(pts[:,0]) - np.min(pts[:,0])
            h_eye = np.max(pts[:,1]) - np.min(pts[:,1])

            # Expansion proportionnelle
            w_eye *= (1.0 + expand_w)
            h_eye *= (1.0 + expand_h)

            mask_np = np.zeros((H, W), dtype=np.uint8)
            center = (int(cx), int(cy))
            axes = (max(1, int(w_eye/2)), max(1, int(h_eye/2)))
            cv2.ellipse(mask_np, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

            mask[b,0] = torch.from_numpy(mask_np/255.0).to(self.device)

        # Feather interne/externe pour adoucir
        mask = feather_inside_strict(mask, radius=2, blur_kernel=3, sigma=0.8)
        mask = feather_outside_only_alpha(mask, radius=2, sigma=1.2)

        if debug and debug_dir is not None and frame_counter == 0:
            os.makedirs(debug_dir, exist_ok=True)
            save_path = os.path.join(debug_dir, f"left_eye_mask_{frame_counter:05d}.png")
            mask_img = (mask[0,0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(save_path)
            print(f"[DEBUG] Left eye mask saved: {save_path}")

        return mask


    def create_right_eye_mask(self, H: int, W: int, debug=False, debug_dir=None, frame_counter=0, expand_w=0.3, expand_h=0.3):
        """
        Masque dynamique pour l’œil droit, ovale réaliste.
        """

        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        for b in range(self.B):
            points = [self.get_point(14)[b].cpu().numpy()]  # right_eye keypoints
            pts = np.array([[p[0]*(W-1), p[1]*(H-1)] for p in points], dtype=np.float32)

            if len(pts) == 0:
                continue

            cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
            w_eye = np.max(pts[:,0]) - np.min(pts[:,0])
            h_eye = np.max(pts[:,1]) - np.min(pts[:,1])

            # Expansion proportionnelle
            w_eye *= (1.0 + expand_w)
            h_eye *= (1.0 + expand_h)

            mask_np = np.zeros((H, W), dtype=np.uint8)
            center = (int(cx), int(cy))
            axes = (max(1, int(w_eye/2)), max(1, int(h_eye/2)))
            cv2.ellipse(mask_np, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

            mask[b,0] = torch.from_numpy(mask_np/255.0).to(self.device)

        # Feather interne/externe pour adoucir
        mask = feather_inside_strict(mask, radius=2, blur_kernel=3, sigma=0.8)
        mask = feather_outside_only_alpha(mask, radius=2, sigma=1.2)

        if debug and debug_dir is not None and frame_counter == 0:
            os.makedirs(debug_dir, exist_ok=True)
            save_path = os.path.join(debug_dir, f"right_eye_mask_{frame_counter:05d}.png")
            mask_img = (mask[0,0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(save_path)
            print(f"[DEBUG] Right eye mask saved: {save_path}")

        return mask
    #-----------------------------------------------------------------------------------------------------------------------------------
    def create_mouth_mask(
        self,
        H: int,
        W: int,
        debug=False,
        debug_dir=None,
        frame_counter=0,
        scale=8
    ):

        # Valeurs par défaut dynamiques
        expand_w = self.get_bouche_expand_w()
        expand_h = self.get_bouche_expand_h()
        # =========================================================
        # 🔥 1. MASK VIA NOUVELLE FONCTION (CORE)
        # =========================================================
        mask = self.get_mouth_region(
            H=H,
            W=W,
            device=self.device,
            debug=debug,
            debug_dir=debug_dir,
            frame_counter=frame_counter
        )

        # =========================================================
        # 🔹 2. EXTRACTION POINTS (pour tracking / debug)
        # =========================================================
        mouth_points_batch = []

        try:
            points_dict = self.estimate_missing_facial_points()
        except Exception as e:
            print(f"[WARN] create_mouth_mask: estimation failed ({e})")
            return mask, mouth_points_batch

        required_keys = ['mouth_left', 'mouth_right', 'mouth_left_c', 'mouth_right_c', 'mouth_top', 'mouth_bottom']

        if not all(k in points_dict for k in required_keys):
            return mask, mouth_points_batch


        if points_dict['mouth_left'] is not None and points_dict['mouth_right'] is not None:
            ml = points_dict['mouth_left']
            mr = points_dict['mouth_right']
            mt = points_dict['mouth_top']
            mb = points_dict['mouth_bottom']

            for b in range(self.B):

                if torch.isnan(ml[b]).any():
                    continue

                mouth_points_batch.append({
                    'mouth_center': ((ml[b] + mr[b]) * 0.5).detach().cpu().numpy(),
                    'mouth_left': ml[b].detach().cpu().numpy(),
                    'mouth_right': mr[b].detach().cpu().numpy(),
                    'mouth_top': mt[b].detach().cpu().numpy(),
                    'mouth_bottom': mb[b].detach().cpu().numpy(),
                })

        else:
            ml = points_dict['mouth_left_c']
            mr = points_dict['mouth_right_c']
            mt = points_dict['mouth_top']
            mb = points_dict['mouth_bottom']

            for b in range(self.B):

                if torch.isnan(ml[b]).any():
                    continue

                mouth_points_batch.append({
                    'mouth_center': ((ml[b] + mr[b]) * 0.5).detach().cpu().numpy(),
                    'mouth_left_c': ml[b].detach().cpu().numpy(),
                    'mouth_right_c': mr[b].detach().cpu().numpy(),
                    'mouth_top': mt[b].detach().cpu().numpy(),
                    'mouth_bottom': mb[b].detach().cpu().numpy(),
                })

        # =========================================================
        # 🔹 3. FEATHER FINAL (léger uniquement)
        # =========================================================
        try:
            mask = feather_inside_strict(mask, radius=1, blur_kernel=3, sigma=0.8) #R 2 ->
            mask = feather_outside_only_alpha(mask, radius=1, sigma=1.2) #R 3 -> 1
        except Exception as e:
            print(f"[WARN] create_mouth_mask: feather failed ({e})")

        # =========================================================
        # 🔹 4. DEBUG VISUEL
        # =========================================================
        if debug and debug_dir is not None and frame_counter == 0:
            save_debug_mask_scale( mask=mask, debug_dir=debug_dir, frame_counter=frame_counter, name="mouth_mask", scale=scale)



        return mask.clamp(0,1), mouth_points_batch
    #-----------------------------------------------------------------------------------------------------------------------------------
    def create_mouth_corners_mask(
        self,
        H: int,
        W: int,
        debug=False,
        debug_dir=None,
        frame_counter=0,
        expand_w=0.40,
        expand_h=0.25,
        temporal_smooth=0.8,
        min_size=6
    ):


        device = self.device
        B = self.B

        mask = torch.zeros((B, 1, H, W), device=device)
        corners_points_batch = []

        # =========================================================
        # 🔹 1. POINTS RÉELS
        # =========================================================
        try:
            points_dict = self.estimate_missing_facial_points()
        except Exception as e:
            print(f"[WARN] mouth_corners: estimation failed ({e})")
            return mask, corners_points_batch

        if 'mouth_left_c' not in points_dict or 'mouth_right_c' not in points_dict:
            return mask, corners_points_batch

        ml = points_dict['mouth_left_c']
        mr = points_dict['mouth_right_c']

        # 🔥 mémoire temporelle
        if not hasattr(self, "_mouth_corners_state"):
            self._mouth_corners_state = {}

        # meshgrid global
        Y, X = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )

        for b in range(B):

            if torch.isnan(ml[b]).any():
                continue

            left  = ml[b]
            right = mr[b]

            # =========================================================
            # 🔹 TEMPORAL SMOOTH
            # =========================================================
            if b in self._mouth_corners_state:
                prev = self._mouth_corners_state[b]
                left  = temporal_smooth * prev["left"]  + (1 - temporal_smooth) * left
                right = temporal_smooth * prev["right"] + (1 - temporal_smooth) * right

            # =========================================================
            # 🔹 DIMENSIONS ADAPTIVES
            # =========================================================
            mouth_width = torch.abs(right[0] - left[0])

            rx = max(min_size, int(mouth_width * W * 0.25 * (1 + expand_w)))
            ry = max(min_size, int(mouth_width * H * 0.15 * (1 + expand_h)))

            # =========================================================
            # 🔹 PIXELS
            # =========================================================
            lx = left[0]  * (W - 1)
            ly = left[1]  * (H - 1)

            rxp = right[0] * (W - 1)
            ryp = right[1] * (H - 1)

            # =========================================================
            # 🔥 MASQUE GAUSSIEN DOUBLE
            # =========================================================
            dist_left = ((X - lx)**2)/(rx**2 + 1e-6) + ((Y - ly)**2)/(ry**2 + 1e-6)
            dist_right = ((X - rxp)**2)/(rx**2 + 1e-6) + ((Y - ryp)**2)/(ry**2 + 1e-6)

            ellipse_left  = torch.exp(-dist_left * 2.5)
            ellipse_right = torch.exp(-dist_right * 2.5)

            combined = torch.maximum(ellipse_left, ellipse_right)

            mask[b,0] = combined

            # =========================================================
            # 🔹 SAVE POINTS
            # =========================================================
            corners_points_batch.append({
                'mouth_left_c': left.detach().cpu().numpy(),
                'mouth_right_c': right.detach().cpu().numpy(),
            })

            # save state
            self._mouth_corners_state[b] = {
                "left": left.detach(),
                "right": right.detach()
            }

            # =========================================================
            # 🔹 DEBUG
            # =========================================================
            if debug:
                print(f"[DEBUG][MOUTH CORNERS]")
                print(f" left: {left.tolist()}")
                print(f" right: {right.tolist()}")
                print(f" rx/ry: {rx}/{ry}")

        # =========================================================
        # 🔹 FEATHER FINAL (léger)
        # =========================================================
        try:
            mask = feather_inside_strict(mask, radius=2, blur_kernel=3, sigma=0.8)
            mask = feather_outside_only_alpha(mask, radius=2, sigma=1.0)
        except Exception as e:
            print(f"[WARN] mouth_corners: feather failed ({e})")

        mask = mask.clamp(0,1)

        # =========================================================
        # 🔹 DEBUG IMAGE
        # =========================================================
        if debug and debug_dir is not None and frame_counter == 0:
            os.makedirs(debug_dir, exist_ok=True)

            mask_img = (mask[0,0].detach().cpu().numpy() * 255).astype(np.uint8)

            save_path = os.path.join(debug_dir, f"mouth_corners_mask_{frame_counter:05d}.png")
            Image.fromarray(mask_img).save(save_path)

            print(f"[DEBUG] Mouth corners mask saved: {save_path}")

        return mask, corners_points_batch
    #-----------------------------------------------------------------------------------------------------------------------------------
    def create_face_mask(self, H: int, W: int, debug=False, debug_dir=None, frame_counter=0):
        """
        Masque facial dynamique et professionnel, en forme ovale réaliste, incluant bouche.
        """
        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        # Masque bouche
        mouth_mask = self.get_mouth_region(
            H, W, device=self.device, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter)

        for b in range(self.B):
            # Points clés du visage (yeux, nez, oreilles)
            points = [
                self.get_point(14)[b].cpu().numpy(),  # right_eye
                self.get_point(15)[b].cpu().numpy(),  # left_eye
                self.get_point(0)[b].cpu().numpy(),   # nose
                self.get_point(16)[b].cpu().numpy(),  # right_ear
                self.get_point(17)[b].cpu().numpy(),   # left_ear
                self.get_point(52)[b].cpu().numpy(),  #  front_left_1
                self.get_point(53)[b].cpu().numpy(),  #  front_left_2
                self.get_point(54)[b].cpu().numpy(),   #  front_m
                self.get_point(55)[b].cpu().numpy(),  #  front_right_1
                self.get_point(56)[b].cpu().numpy(),   #  front_right_2
                self.get_point(40)[b].cpu().numpy(),    #mouth_left 40
                self.get_point(41)[b].cpu().numpy()     #mouth_right 41
            ]
            pts = np.array([[p[0]*(W-1), p[1]*(H-1)] for p in points], dtype=np.float32)

            # Centre et dimensions de l'ovale
            cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
            w_face = np.max(pts[:,0]) - np.min(pts[:,0])
            h_face = np.max(pts[:,1]) - np.min(pts[:,1])

            # Ajustement pour un ovale réaliste (plus haut que large)
            w_face *= 1.1  # élargir légèrement
            h_face *= 1.55   # hauteur accentuée

            # Création masque ovale
            mask_np = np.zeros((H, W), dtype=np.uint8)
            center = (int(cx), int(cy))
            axes = (int(w_face/2), int(h_face/2))
            cv2.ellipse(mask_np, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

            # Combiner avec le masque bouche
            mask[b,0] = torch.maximum(torch.from_numpy(mask_np/255.0).to(self.device), mouth_mask[b,0])

        # Feather interne et externe pour adoucir
        mask = feather_inside_strict(mask, radius=3, blur_kernel=3, sigma=1.0)
        mask = feather_outside_only_alpha(mask, radius=3, sigma=1.5)

        # Debug

        if debug and debug_dir is not None and frame_counter == 0:
            os.makedirs(debug_dir, exist_ok=True)
            save_path = os.path.join(debug_dir, f"face_mask_pro_{frame_counter:05d}.png")
            mask_img = (mask[0,0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(save_path)
            print(f"[DEBUG] Professional face mask saved: {save_path}")

        return mask

        # -------------------- Debug --------------------
        if debug and debug_dir is not None:
            save_debug_mask(mask, H, W, debug_dir, frame_counter, prefix="face_mask_")

        return mask

# ----------------------------------------------------------------------------------------------------------------------------------
# Pose PoseAnimator en DEV
# ----------------------------------------------------------------------------------------------------------------------------------

class PoseAnimator:
    def __init__(self, pose: Pose, latent_h: int, latent_w: int):
        self.pose = pose
        self.H = latent_h
        self.W = latent_w

    # ----------------- Préparer les masques -----------------
    def prepare_masks(self, debug=False, debug_dir=None, frame_counter=0):
        # Masque torse
        self.torso_mask = self.pose.create_upper_body_mask(
            H=self.H, W=self.W,
            expand_w=0.9, shrink_h=0.65,
            debug=debug, debug_dir=debug_dir, frame_counter=frame_counter
        )

        # Masque visage dynamique
        self.face_mask = self.pose.create_face_mask(
            H=self.H, W=self.W,
            debug=debug, debug_dir=debug_dir, frame_counter=frame_counter
        )

        # Masque cheveux
        self.hair_mask = self.pose.create_hair_mask(
            H=self.H, W=self.W,
            debug=debug, debug_dir=debug_dir, frame_counter=frame_counter
        )

    # ----------------- Calcul des deltas -----------------
    def compute_deltas(self, torso_scale=1.0, face_scale=0.3, x_factor=0.3):
        # Torse
        self.pose.compute_torso_delta(latent_h=self.H, latent_w=self.W)
        self.torso_delta = self.pose.delta.clone()
        self.torso_delta[:,0] *= x_factor  # limiter le déplacement X

        # Face : micro-mouvements
        face_center = (self.pose.get_point(14) + self.pose.get_point(15) + self.pose.get_point(0)) / 3.0
        face_delta = face_center * face_scale
        face_delta = torch.tanh(face_delta * 2.0) * 0.3
        self.face_delta = face_delta.clone()
        self.face_delta[:,0] *= x_factor  # réduire X

    # ----------------- Appliquer les deltas sur le latent -----------------
    def apply_to_latent(self, latent: torch.Tensor):
        """
        latent: [B, C, H, W]
        Retourne latent avec torse + visage animés
        """

        # Copier pour éviter d'écraser
        out = latent.clone()

        # Appliquer delta torse
        torso_mask = self.torso_mask
        out = self.grid_warp(out, torso_mask, self.torso_delta)

        # Appliquer delta visage
        face_mask = self.face_mask
        out = self.grid_warp(out, face_mask, self.face_delta)

        return out

    # ----------------- Grid warp localisé -----------------
    @staticmethod
    def grid_warp(latent, mask, delta):
        """
        Applique un déplacement (delta) uniquement sur la zone du mask
        """
        B, C, H, W = latent.shape

        # Créer grid [B,H,W,2]
        yy, xx = torch.meshgrid(torch.arange(H, device=latent.device),
                                torch.arange(W, device=latent.device),
                                indexing='ij')
        grid = torch.stack([xx.float(), yy.float()], dim=-1)  # [H,W,2]
        grid = grid.unsqueeze(0).repeat(B,1,1,1)  # [B,H,W,2]

        # Ajouter delta multiplié par mask
        mask_expand = mask.permute(0,2,3,1)  # [B,H,W,1]
        delta_expand = delta.unsqueeze(1).unsqueeze(1)  # [B,1,1,2]
        grid = grid + delta_expand * mask_expand

        # Normaliser grid entre -1 et 1
        grid_norm = grid.clone()
        grid_norm[...,0] = 2.0 * grid[...,0] / (W-1) - 1.0
        grid_norm[...,1] = 2.0 * grid[...,1] / (H-1) - 1.0

        # Sample latent
        out = torch.nn.functional.grid_sample(latent, grid_norm, align_corners=True)
        return out

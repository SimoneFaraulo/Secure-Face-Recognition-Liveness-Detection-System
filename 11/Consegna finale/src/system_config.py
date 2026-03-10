import os
import torch


class Config:
    BASE_DIR = "/content/drive/MyDrive/11/"
    GALLERY_EMBEDDINGS = os.path.join(BASE_DIR, "Consegna parziale/features_embeddings_gr_11_copy")
    GALLERY_LABELS = os.path.join(BASE_DIR, "Consegna parziale/features_labels_gr_11_copy")
    LIVENESS_WEIGHTS = os.path.join(BASE_DIR, "Consegna parziale/models/best_weights_spoof_gr_11.pth")
    FACENET_WEIGHTS = None

    T_i = 0.7898
    T_v = 0.7909
    T_f = 0.6459

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    MAX_ATTEMPTS = 3
    BLUR_THRESH = 80.0
    CONTRAST = 40.0
    MIN_FACE_SIZE = 450
    SYMMETRY_THRESH = 0.3
    PITCH_RANGE = (0.35, 1.80)
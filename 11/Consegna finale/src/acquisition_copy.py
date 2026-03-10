import cv2
import numpy as np
import time
import torch
from facenet_pytorch import MTCNN
import os

class BiometricAcquisition:
    """
    Acquisizione biometrica con controllo posa 3D (Yaw e Pitch) usando landmarks 2D.
    """

    def __init__(self,
                 max_attempts=3,
                 blur_threshold=100.0,
                 contrast_threshold=20.0,
                 brightness_range=(40, 220),
                 min_face_size=100,
                 symmetry_threshold=0.20,
                 pitch_range=(0.50, 1.50)
                 ):
        self.max_attempts = max_attempts
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold
        self.brightness_range = brightness_range
        self.min_face_size = min_face_size
        self.symmetry_threshold = symmetry_threshold
        self.pitch_range = pitch_range

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.detector = MTCNN(
            keep_all=True,
            min_face_size=self.min_face_size,
            device=self.device
        )
        print(f"[LIB] Detector inizializzato su {self.device}. Controllo Yaw + Pitch attivo.")

    def _compute_metrics(self, img):
        if img is None or img.size == 0: return 0.0, 0.0, 0.0
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blur = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        contrast = gray_img.std()
        brightness = gray_img.mean()
        return blur, contrast, brightness

    def _check_head_pose(self, landmarks):
        eye_l = landmarks[0]
        eye_r = landmarks[1]
        nose = landmarks[2]
        mouth_l = landmarks[3]
        mouth_r = landmarks[4]

        dist_l_x = abs(nose[0] - mouth_l[0])
        dist_r_x = abs(mouth_r[0] - nose[0])

        if dist_l_x == 0 or dist_r_x == 0: return False, 1.0, False, 0.0

        yaw_diff = abs(dist_l_x - dist_r_x)
        yaw_total = dist_l_x + dist_r_x
        yaw_score = yaw_diff / yaw_total if yaw_total > 0 else 1.0

        is_frontal_yaw = yaw_score <= self.symmetry_threshold

        mid_eyes = (eye_l + eye_r) / 2
        mid_mouth = (mouth_l + mouth_r) / 2

        dist_eyes_nose = np.linalg.norm(mid_eyes - nose)
        dist_nose_mouth = np.linalg.norm(nose - mid_mouth)

        if dist_nose_mouth == 0: return is_frontal_yaw, yaw_score, False, 999.0

        pitch_ratio = dist_eyes_nose / dist_nose_mouth

        if pitch_ratio < self.pitch_range[0]:
            pitch_status = "ALTO"
            is_frontal_pitch = False
        elif pitch_ratio > self.pitch_range[1]:
            pitch_status = "BASSO"
            is_frontal_pitch = False
        else:
            pitch_status = "OK"
            is_frontal_pitch = True

        return is_frontal_yaw, yaw_score, is_frontal_pitch, pitch_ratio, pitch_status

    def _extract_and_score_face(self, img):
        if img is None: return None, {}, "Immagine non valida/Null."

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:

            boxes, probs, points = self.detector.detect(img_rgb, landmarks=True)
        except RuntimeError as e:

            if "torch.cat" in str(e):
                return None, {}, "Nessun volto rilevato (MTCNN Error)."
            else:
                raise e
        except Exception as e:
            return None, {}, f"Errore generico detection: {str(e)}"

        if boxes is None or len(boxes) == 0:
            return None, {}, "Nessun volto rilevato."
        
        if len(boxes) > 1:
            pass

        box = boxes[0]
        confidence = probs[0] if probs is not None else 0.0
        landmarks = points[0] if points is not None else None

        if landmarks is None:
             return None, {}, "Landmarks non rilevati."

        x1, y1, x2, y2 = [int(b) for b in box]

        img_h, img_w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        face_roi = img[y1:y2, x1:x2]
        if face_roi.size == 0: return None, {}, "ROI vuota."

        blur, contrast, brightness = self._compute_metrics(face_roi)
        is_yaw_ok, yaw_score, is_pitch_ok, pitch_ratio, pitch_status_str = self._check_head_pose(landmarks)

        metrics = {
            'blur': blur,
            'contrast': contrast,
            'brightness': brightness,
            'confidence': confidence,
            'yaw_score': yaw_score,
            'pitch_ratio': pitch_ratio,
            'is_yaw_ok': is_yaw_ok,
            'is_pitch_ok': is_pitch_ok,
            'pitch_str': pitch_status_str
        }

        pose_msg = []
        if not is_yaw_ok: pose_msg.append(f"GIRATO(score {yaw_score:.2f})")
        if not is_pitch_ok: pose_msg.append(f"SGUARDO {pitch_status_str}(ratio {pitch_ratio:.2f})")
        if is_yaw_ok and is_pitch_ok: pose_msg.append("FRONTALE")

        msg = f"Volto {' '.join(pose_msg)}: B={blur:.0f}, C={contrast:.0f}"
        return face_roi, metrics, msg

    def _is_quality_pass(self, metrics):
        """
        Versione CORRETTA che restituisce SEMPRE una tupla (bool, str).
        """
        reasons = []

        if not metrics.get('confidence', 0) > 0.90:
             reasons.append(f"Confidence bassa ({metrics.get('confidence', 0):.2f})")

        if not metrics['is_yaw_ok']:
            reasons.append(f"Testa girata orizzontalmente ({metrics['yaw_score']:.2f})")

        if not metrics['is_pitch_ok']:
            reasons.append(f"Testa girata verticalmente ({metrics['pitch_str']})")

        if metrics['blur'] < self.blur_threshold:
            reasons.append(f"Sfocato ({metrics['blur']:.1f})")
            
        if metrics['contrast'] < self.contrast_threshold:
            reasons.append(f"Contrasto basso ({metrics['contrast']:.1f})")
            
        if not (self.brightness_range[0] <= metrics['brightness'] <= self.brightness_range[1]):
            reasons.append(f"Problema luce ({metrics['brightness']:.1f})")

        if reasons:
            return False, " | ".join(reasons)
        
        return True, "OK"

    def acquire_best_sample(self, capture_function, filename='face.jpg'):
        """Metodo per acquisizione da Webcam"""
        best_face = None
        best_score = -1.0
        print(f"Inizio procedura (Yaw Th: {self.symmetry_threshold}, Pitch Range: {self.pitch_range})...")

        for i in range(1, self.max_attempts + 1):
            print(f"\n--- Tentativo {i}/{self.max_attempts} ---")
            img = capture_function()
            if img is None: continue

            face, metrics, msg = self._extract_and_score_face(img)
            print(msg)

            if face is not None:
                is_valid, reason = self._is_quality_pass(metrics)

                if is_valid:
                    current_score = metrics['blur'] * metrics['contrast']
                    if current_score > best_score:
                        best_score = current_score
                        best_face = face
                        cv2.imwrite(filename, face)
                        return face, time.time()
                else:
                    print(f"  -> SCARTATO: {reason}")

            time.sleep(1.5)

        return None, time.time()

    def acquire_from_file(self, image_path):
        """Metodo per acquisizione da File (Test su Cartella)"""
        if not os.path.exists(image_path):
            return None, False, {}, f"File non trovato: {image_path}"

        img = cv2.imread(image_path)
        if img is None:
            return None, False, {}, "Impossibile leggere l'immagine."

        face_roi, metrics, msg = self._extract_and_score_face(img)
        
        if face_roi is None:
            return None, False, {}, f"Nessun volto rilevato: {msg}"

        is_valid, reason = self._is_quality_pass(metrics)
        
        full_msg = f"{msg} -> Esito: {'APPROVATO' if is_valid else 'RIFIUTATO'}"
        if not is_valid:
            full_msg += f" ({reason})"

        return face_roi, is_valid, metrics, full_msg
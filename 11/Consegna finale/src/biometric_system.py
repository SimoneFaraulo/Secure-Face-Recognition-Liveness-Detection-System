import os
import cv2
import torch
import numpy as np
import time
import psutil
import sys
from PIL import Image
from torchvision import transforms
from IPython.display import clear_output

from acquisition import BiometricAcquisition
from feature_extractor import FaceRecognitionSystem
from liveness_detector import SpoofingModel
from system_config import *


class IntegratedBiometricSystem:
    def __init__(self, config):
        self.cfg = config
        self.device = config.DEVICE

        print(">>> Inizializzazione Sistema Biometrico...")

        self.acquisition = BiometricAcquisition(
            max_attempts=self.cfg.MAX_ATTEMPTS,
            blur_threshold=self.cfg.BLUR_THRESH,
            min_face_size=self.cfg.MIN_FACE_SIZE,
            symmetry_threshold=self.cfg.SYMMETRY_THRESH,
            pitch_range=self.cfg.PITCH_RANGE,
            contrast_threshold= self.cfg.CONTRAST,
        )

        self.face_system = FaceRecognitionSystem(device=self.device, weights_path=self.cfg.FACENET_WEIGHTS)

        if os.path.exists(self.cfg.GALLERY_EMBEDDINGS + ".npy"):
            self.face_system.load_existing_gallery(self.cfg.GALLERY_EMBEDDINGS, self.cfg.GALLERY_LABELS)
        else:
            print("ATTENZIONE: Nessuna Gallery trovata. Necessario Enrollment.")

        self.liveness_model = SpoofingModel(pretrained=False).to(self.device)

        if os.path.exists(self.cfg.LIVENESS_WEIGHTS):
            print(f"--- Caricamento pesi Liveness da: {self.cfg.LIVENESS_WEIGHTS} ---")
            checkpoint = torch.load(self.cfg.LIVENESS_WEIGHTS, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print(f"   >>> Rilevato CheckpointManager (Best Loss: {checkpoint.get('best_loss', 'N/A')})")
                self.liveness_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("   >>> Rilevato formato pesi standard.")
                self.liveness_model.load_state_dict(checkpoint)
            self.liveness_model.eval()
            print(">>> Modello Liveness pronto.")
        else:
            print(f"ERRORE: Pesi Liveness non trovati in {self.cfg.LIVENESS_WEIGHTS}")

        self.liveness_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _print_menu_text(self):
        """Helper per stampare il menu"""
        print("\n--- SELEZIONA MODALITÀ ---")
        print("1. Enrollment (Registra nuovo utente)")
        print("2. Identification (Chi sono? - Open Set)")
        print("3. Verification (Io sono l'utente X, verificami)")

    def _get_valid_input(self, prompt, cast_type=str, valid_options=None, reprint_callback=None):
        """
        Gestisce l'input utente. In caso di errore, stampa un avviso e richiede l'input
        nella riga successiva (senza cancellare nulla).
        """
        while True:
            try:
                user_input = input(prompt)

                value = cast_type(user_input)

                if valid_options:
                    check_val = value.lower() if isinstance(value, str) else value
                    if check_val not in valid_options:
                        raise ValueError("Opzione non valida")

                return value
            except (ValueError, IndexError):
                print(f">>> Valore non valido. Riprova.")

    def check_liveness(self, face_img):
        """Esegue il controllo di liveness e ritorna (Score, Decisione)"""
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        img_tensor = self.liveness_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.liveness_model(img_tensor)
            score = torch.sigmoid(output).item()
        return score, score > self.cfg.T_f

    def extract_features_only(self, face_img):
        """Estrae solo il vettore delle features usando il modello FaceNet caricato"""
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        img_tensor = self.face_system.single_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.face_system.resnet(img_tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()

    def enroll_user(self, face_img, user_id):
        """Aggiunge un nuovo utente alla Gallery in memoria e su disco"""
        features = self.extract_features_only(face_img)
        if self.face_system.gallery_embeddings is None:
            self.face_system.gallery_embeddings = features
            self.face_system.gallery_labels = np.array([user_id])
        else:
            self.face_system.gallery_embeddings = np.concatenate([self.face_system.gallery_embeddings, features],
                                                                 axis=0)
            self.face_system.gallery_labels = np.concatenate([self.face_system.gallery_labels, np.array([user_id])],
                                                             axis=0)
        self.face_system._load_gallery_to_gpu()
        np.save(self.cfg.GALLERY_EMBEDDINGS + ".npy", self.face_system.gallery_embeddings)
        np.save(self.cfg.GALLERY_LABELS + ".npy", self.face_system.gallery_labels)
        print(f"Utente {user_id} registrato con successo!")

    def identify_user(self, face_img):
        """Open-set Identification: Ritorna lista ordinata di match o Reject"""
        if self.face_system.gallery_embeddings_tensor is None: return []
        probe_tensor = torch.tensor(self.extract_features_only(face_img)).to(self.device)
        dists = torch.cdist(probe_tensor, self.face_system.gallery_embeddings_tensor, p=2).squeeze(0)
        mask = dists < self.cfg.T_i
        valid_indices = torch.nonzero(mask).flatten()
        if len(valid_indices) == 0: return []
        results = []
        for idx in valid_indices:
            idx_int = idx.item()
            results.append((self.face_system.gallery_labels[idx_int], dists[idx_int].item()))
        results.sort(key=lambda x: x[1])
        return results

    def verify_user(self, face_img, claimed_id):
        """Verification: confronta SOLO con l'ID dichiarato"""
        indices = np.where(self.face_system.gallery_labels == claimed_id)[0]
        if len(indices) == 0: return float('inf'), "USER_NOT_FOUND"
        target_embeddings = self.face_system.gallery_embeddings_tensor[indices]
        probe_embedding = torch.tensor(self.extract_features_only(face_img)).to(self.device)
        min_dist = torch.min(torch.cdist(probe_embedding, target_embeddings, p=2)).item()
        return min_dist, "MATCH" if min_dist < self.cfg.T_v else "REJECT"

    def _print_memory_usage(self):
        """Stampa l'utilizzo di memoria RAM e GPU."""
        print("\n--- STATISTICHE MEMORIA ---")
        process = psutil.Process(os.getpid())
        ram_usage_gb = process.memory_info().rss / (1024 ** 3)
        print(f"RAM Utilizzata (Processo): {ram_usage_gb:.2f} GB")

        if torch.cuda.is_available():
            gpu_usage_gb = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            print(f"VRAM Utilizzata (GPU): {gpu_usage_gb:.2f} GB")
            print(f"Totale Stimato: {ram_usage_gb + gpu_usage_gb:.2f} GB")
        else:
            print(f"Totale Stimato: {ram_usage_gb:.2f} GB")

        if ram_usage_gb > 4.0:
            print("ATTENZIONE: Memoria > 4GB")
        else:
            print(">>> Requisito Memoria (< 4GB) RISPETTATO.")

    def run_pipeline(self, capture_func):
        self._print_menu_text()

        mode = self._get_valid_input(
            "Inserisci numero (1/2/3): ",
            valid_options=['1', '2', '3'],
            reprint_callback=None
        )

        liveness_input = self._get_valid_input("Attivare Liveness Detection? (s/n): ", valid_options=['s', 'n'])
        use_liveness = (liveness_input.lower() == 's')

        claimed_id_input = None
        if mode == '3':
            claimed_id_input = self._get_valid_input(">>> Inserisci ID utente da verificare (claimed ID): ",
                                                     cast_type=int)

        print("\n>>> Avvio Acquisizione (Guardare in camera)...")
        face_img, acq_time = self.acquisition.acquire_best_sample(capture_func)

        if face_img is None:
            print("Acquisizione fallita. Riprovare.")
            return

        print(">>> Acquisizione completata. Avvio timer elaborazione...")
        start_processing_time = time.time()

        if use_liveness:
            print(">>> Eseguo controllo Liveness...")
            liveness_score, is_live = self.check_liveness(face_img)
            status = "LIVE" if is_live else "FAKE"
            print(f"   -> Liveness Score: {liveness_score:.4f} ({status})")

            if not is_live:
                print(
                    f"!!! ACCESSO NEGATO: Rilevato Replay Attack (Score {liveness_score:.2f} < Soglia {self.cfg.T_f}) !!!")
                self._print_memory_usage()
                return
        else:
            print(">>> Liveness Detection: DISATTIVATA")

        if mode == '1':
            user_id = self._get_valid_input("Inserisci ID numerico per il nuovo utente: ", cast_type=int)
            self.enroll_user(face_img, user_id)

        elif mode == '2':
            matches = self.identify_user(face_img)
            print(f"\nRISULTATO IDENTIFICATION (Soglia Ti={self.cfg.T_i}):")
            if not matches:
                print(">>> REJECT: Nessun utente trovato sopra la soglia.")
            else:
                print(f"Trovati {len(matches)} possibili match:")
                for i, (uid, dist) in enumerate(matches):
                    print(f"   {i + 1}. Utente {uid} (Distanza: {dist:.4f})")
                print(f">>> IDENTIFICATO COME: Utente {matches[0][0]}")

        elif mode == '3':
            if claimed_id_input is not None:
                dist, decision = self.verify_user(face_img, claimed_id_input)
                print(
                    f"\nRISULTATO VERIFICATION:\nUtente Dichiarato: {claimed_id_input}\nDistanza Minima: {dist:.4f}\nDecisione: {decision}")

        processing_time = time.time() - start_processing_time
        print(f"\nTempo Elaborazione (Post-Acquisizione): {processing_time:.2f} secondi.")

        self._print_memory_usage()
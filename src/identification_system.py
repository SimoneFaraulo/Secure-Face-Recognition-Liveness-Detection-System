import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from list_dataset import ListDataset
from feature_extractor import FaceRecognitionSystem


class IdentificationSystem(FaceRecognitionSystem):
    """
    Gestisce SOLO l'Identification (Open-Set).
    Calcola la soglia Ti basandosi sulla distribuzione degli Impostori.
    """

    def compute_identification_threshold(self, probes_root_dir, val_split=0.4, target_far=0.05, random_seed=42):
        """
        Calcola Ti usando il percentile della distribuzione degli impostori.
        Target FAR = 0.05 significa che accettiamo il 5% di falsi positivi.
        """
        if self.gallery_embeddings is None:
            print("ERRORE: Carica prima la Gallery!")
            return None

        if self.gallery_embeddings_tensor is None:
            self._load_gallery_to_gpu()

        print(f"\n--- Inizio Calcolo Soglia Identification (Ti) ---")
        print(f"Target FAR richiesto: {target_far * 100}%")

        all_samples = []
        all_labels = []

        if not os.path.exists(probes_root_dir):
            print(f"Errore: path {probes_root_dir} non esiste")
            return None

        subjects = sorted([d for d in os.listdir(probes_root_dir) if os.path.isdir(os.path.join(probes_root_dir, d))])

        for subj_dir in subjects:
            subj_path = os.path.join(probes_root_dir, subj_dir)
            if "subject_" in subj_dir:
                try:
                    label = int(subj_dir.split('_')[1])
                    files = sorted(os.listdir(subj_path))
                    for fname in files:
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            all_samples.append((os.path.join(subj_path, fname), label))
                            all_labels.append(label)
                except:
                    continue

        if not all_samples: return None

        X_val, _, _, _ = train_test_split(
            all_samples, all_labels,
            test_size=(1 - val_split),
            stratify=all_labels,
            random_state=random_seed
        )

        print(f"Validation Set per Ti: {len(X_val)} immagini (Impostori simulati)")

        val_dataset = ListDataset(X_val, transform=self.single_transform)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

        val_embeddings = []
        val_labels_list = []

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(self.device)
                emb = self.resnet(imgs)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                val_embeddings.append(emb)
                val_labels_list.append(lbls)

        if not val_embeddings:
            print("Errore: Nessuna features estratta.")
            return None

        val_emb_tensor = torch.cat(val_embeddings)
        val_lbl_np = np.concatenate([l.numpy() for l in val_labels_list])

        dists = torch.cdist(val_emb_tensor, self.gallery_embeddings_tensor, p=2)
        dists_np = dists.cpu().numpy()
        gal_lbl_np = self.gallery_labels

        unknown_scores = []
        known_scores = []

        for i, true_label in enumerate(val_lbl_np):
            mask_same = (gal_lbl_np == true_label)
            if np.any(mask_same):
                known_scores.append(np.min(dists_np[i, mask_same]))

            mask_others = (gal_lbl_np != true_label)
            if np.any(mask_others):
                best_impostor_dist = np.min(dists_np[i, mask_others])
                unknown_scores.append(best_impostor_dist)

        unknown_scores = np.array(unknown_scores)
        known_scores = np.array(known_scores)

        if len(unknown_scores) == 0:
            print("Errore: Impossibile simulare impostori.")
            return None

        ti = np.quantile(unknown_scores, target_far)

        print(f"\n--- RISULTATI CALCOLO SOGLIA ---")
        print(f"Soglia Calcolata (Ti): {ti:.4f}")

        plt.figure(figsize=(10, 6))

        plt.hist(known_scores, bins=30, alpha=0.6, color='green', label='Known Users (Best Match)', density=True)

        plt.hist(unknown_scores, bins=30, alpha=0.6, color='red', label='Unknown/Impostors (Best Match)', density=True)

        plt.axvline(ti, color='blue', linestyle='--', linewidth=3, label=f'Threshold Ti (FAR={target_far * 100}%)')

        plt.xlabel('Best Matching Distance (Euclidean)')
        plt.ylabel('Density')
        plt.title('Open-Set Identification Analysis (Validation Set)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        false_accepts = np.sum(unknown_scores <= ti)
        actual_far = false_accepts / len(unknown_scores)
        true_accepts = np.sum(known_scores <= ti)
        dir_rate = true_accepts / len(known_scores) if len(known_scores) > 0 else 0

        print(f"Verifica FAR: {actual_far * 100:.2f}%")
        print(f"DIR stimato: {dir_rate * 100:.2f}%")

        return ti

    def evaluate_open_set_test(self, test_samples, test_labels, ti):
        if self.gallery_embeddings_tensor is None:
            self._load_gallery_to_gpu()

        print(f"\n--- VALUTAZIONE TEST SET (Open-Set) ---")

        combined_data = list(zip(test_samples, test_labels))
        test_dataset = ListDataset(combined_data, transform=self.single_transform)
        loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

        test_embeddings = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                emb = self.resnet(imgs)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                test_embeddings.append(emb)

        if not test_embeddings:
            print("Nessun dato di test.")
            return

        test_emb_tensor = torch.cat(test_embeddings)
        dists = torch.cdist(test_emb_tensor, self.gallery_embeddings_tensor, p=2)
        dists_np = dists.cpu().numpy()

        test_lbl_np = np.array(test_labels)
        gal_lbl_np = self.gallery_labels

        total_attempts = len(test_lbl_np)

        correct_id = 0
        false_reject = 0
        misidentified = 0
        unknown_accepted = 0
        unknown_rejected = 0

        for i, true_label in enumerate(test_lbl_np):
            best_idx = np.argmin(dists_np[i])
            min_dist = dists_np[i, best_idx]
            pred_id = gal_lbl_np[best_idx]

            if min_dist > ti:
                false_reject += 1
            elif pred_id == true_label:
                correct_id += 1
            else:
                misidentified += 1

            mask_others = (gal_lbl_np != true_label)
            if np.any(mask_others):
                dist_unknown = np.min(dists_np[i, mask_others])
                if dist_unknown <= ti:
                    unknown_accepted += 1
                else:
                    unknown_rejected += 1

        dir_metric = correct_id / total_attempts
        frr_metric = (false_reject + misidentified) / total_attempts
        far_metric = unknown_accepted / total_attempts

        print(f"Soglia Ti usata: {ti:.4f}")
        print("-" * 40)
        print(f"METRICHE ISCRITTI:")
        print(f"  DIR: {dir_metric * 100:.2f}%")
        print(f"  FRR: {frr_metric * 100:.2f}%")
        print("-" * 40)
        print(f"METRICHE SCONOSCIUTI:")
        print(f"  FAR: {far_metric * 100:.2f}%")
        print("=" * 40)
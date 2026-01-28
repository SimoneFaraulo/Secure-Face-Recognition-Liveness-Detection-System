import os
import torch
import numpy as np
import time
import re
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from feature_extractor import FaceRecognitionSystem
from list_dataset import ListDataset


class VerificationSystem(FaceRecognitionSystem):

    def compute_validation_thresholds(self, probes_root_dir, val_split=0.4, random_seed=42):
        if self.gallery_embeddings is None:
            print("ERRORE: Carica prima la Gallery!")
            return None

        if self.gallery_embeddings_tensor is None:
            self._load_gallery_to_gpu()

        print(f"\n--- Inizio Calcolo Soglie (Val Split: {val_split}) ---")

        all_samples = []
        all_labels = []

        def get_subject_id(name):
            match = re.search(r'\d+', name)
            return int(match.group()) if match else 999999

        if not os.path.exists(probes_root_dir):
            print(f"Errore: path {probes_root_dir} non esiste")
            return None

        subjects = sorted([d for d in os.listdir(probes_root_dir) if os.path.isdir(os.path.join(probes_root_dir, d))])

        for subj in subjects:
            subj_path = os.path.join(probes_root_dir, subj)
            label = get_subject_id(subj)
            files = sorted([f for f in os.listdir(subj_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            for f in files:
                all_samples.append((os.path.join(subj_path, f), label))
                all_labels.append(label)

        if not all_samples:
            print("Nessuna immagine trovata nei probes.")
            return None

        X_val, X_test, y_val, y_test = train_test_split(
            all_samples, all_labels,
            test_size=(1 - val_split),
            stratify=all_labels,
            random_state=random_seed
        )

        print(f"Totale Probes: {len(all_samples)}")
        print(f"Validation Set (usato per ROC/Soglia): {len(X_val)}")
        print(f"Test Set (DA SALVARE/USARE DOPO): {len(X_test)}")

        val_dataset = ListDataset(X_val, transform=self.single_transform)
        bs = 32
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=2)

        val_embeddings_list = []
        val_labels_list = []

        print("Estrazione features validation set in corso...")
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(self.device)
                emb = self.resnet(imgs)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                val_embeddings_list.append(emb)
                val_labels_list.append(lbls)

        val_emb_tensor = torch.cat(val_embeddings_list)
        val_lbl_np = np.concatenate([l.numpy() for l in val_labels_list])

        print("Calcolo matrice distanze (Batch)...")
        dists = torch.cdist(val_emb_tensor, self.gallery_embeddings_tensor, p=2)

        dists_np = dists.cpu().numpy()
        gal_lbl_np = self.gallery_labels

        genuine_dists = []
        impostor_dists = []

        for i, probe_label in enumerate(val_lbl_np):
            mask_same = (gal_lbl_np == probe_label)
            if np.any(mask_same):
                genuine_dists.append(np.min(dists_np[i, mask_same]))

            mask_diff = (~mask_same)
            if np.any(mask_diff):
                impostor_dists.append(np.min(dists_np[i, mask_diff]))

        genuine_dists = np.array(genuine_dists)
        impostor_dists = np.array(impostor_dists)

        y_true = np.concatenate([np.zeros(len(genuine_dists)), np.ones(len(impostor_dists))])
        y_scores = np.concatenate([genuine_dists, impostor_dists])

        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)

        fnr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fpr - fnr))
        tv_eer = thresholds[eer_index]
        eer_val = fpr[eer_index]

        print(f"\n=== RISULTATI SOGLIA (Tv) ===")
        print(f"Soglia EER: {tv_eer:.4f} (EER: {eer_val * 100:.2f}%)")

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        plt.scatter(fpr[eer_index], tpr[eer_index], color='red', s=50, zorder=5,
                    label=f'EER Point ({eer_val:.2f}, {1 - eer_val:.2f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Impostori Accettati)')
        plt.ylabel('True Positive Rate (Impostori Rifiutati correttamente)')
        plt.title('Receiver Operating Characteristic (Verification)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()


        return tv_eer

    def evaluate_test_set(self, test_samples, test_labels, tv):
        if self.gallery_embeddings is None:
            print("ERRORE: Gallery non caricata.")
            return

        if self.gallery_embeddings_tensor is None:
            self._load_gallery_to_gpu()

        print(f"\n--- VALUTAZIONE TEST SET ({len(test_samples)} campioni) ---")

        combined_data = list(zip(test_samples, test_labels))

        bs = 32
        test_dataset = ListDataset(combined_data, transform=self.single_transform)
        loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=2)

        test_embeddings_list = []

        print("Estrazione feature Test Set...")
        start_time = time.time()

        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                emb = self.resnet(imgs)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                test_embeddings_list.append(emb)

        test_emb_tensor = torch.cat(test_embeddings_list)
        end_time = time.time()
        print(f"Tempo estrazione: {end_time - start_time:.2f}s")

        dists = torch.cdist(test_emb_tensor, self.gallery_embeddings_tensor, p=2)
        dists_np = dists.cpu().numpy()
        test_labels_np = np.array(test_labels)
        gallery_labels_np = self.gallery_labels

        min_dists_indices = np.argmin(dists_np, axis=1)
        predicted_labels = gallery_labels_np[min_dists_indices]
        correct_identifications = np.sum(predicted_labels == test_labels_np)
        rank_1_acc = correct_identifications / len(test_labels_np)

        genuine_dists = []
        impostor_dists = []

        for i, probe_lbl in enumerate(test_labels_np):
            mask_same = (gallery_labels_np == probe_lbl)
            mask_diff = (~mask_same)

            if np.any(mask_same):
                genuine_dists.append(np.min(dists_np[i, mask_same]))

            if np.any(mask_diff):
                impostor_dists.append(np.min(dists_np[i, mask_diff]))

        genuine_dists = np.array(genuine_dists)
        impostor_dists = np.array(impostor_dists)

        if len(genuine_dists) > 0:
            false_rejects = np.sum(genuine_dists > tv)
            frr = false_rejects / len(genuine_dists)
        else:
            frr = 0.0

        if len(impostor_dists) > 0:
            false_accepts = np.sum(impostor_dists <= tv)
            far = false_accepts / len(impostor_dists)
        else:
            far = 0.0

        print("\n" + "=" * 40)
        print(" RISULTATI FINALI SUL TEST SET")
        print("=" * 40)
        print(f"Soglia usata (Tv): {tv:.4f}")
        print("-" * 40)
        print(f"Rank-1 Identification Rate: {rank_1_acc * 100:.2f}%")
        print("-" * 40)
        print(f"False Acceptance Rate (FAR): {far * 100:.2f}%")
        print(f"False Rejection Rate  (FRR): {frr * 100:.2f}%")
        print("=" * 40)
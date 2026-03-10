import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from list_dataset import ListDataset
import time

try:
    from feature_extractor import FaceRecognitionSystem
except ImportError:
    raise ImportError("Il file 'feature_extractor.py' deve essere nella stessa directory.")


class IdentificationSystem(FaceRecognitionSystem):

    def compute_identification_thresholds(self, probes_root_dir, val_split=0.4, target_fars=None, random_seed=42):
        if target_fars is None:
            target_fars = [0.01, 0.05, 0.1]
        if self.gallery_embeddings is None:
            print("ERRORE: Carica prima la Gallery!")
            return None

        if self.gallery_embeddings_tensor is None:
            self._load_gallery_to_gpu()

        print(f"\n--- Inizio Calcolo Soglie Identification (Ti) ---")
        print(f"Target FARs richiesti: {[f'{x * 100}%' for x in target_fars]}")

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

        print(f"Validation Set per Ti: {len(X_val)} immagini")

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

        calculated_thresholds = {}

        plt.figure(figsize=(12, 6))
        plt.hist(known_scores, bins=30, alpha=0.6, color='green', label='Known Users', density=True)
        plt.hist(unknown_scores, bins=30, alpha=0.6, color='red', label='Unknown/Impostors', density=True)

        print(f"\n--- RISULTATI CALCOLO SOGLIE (Validation) ---")

        colors = ['blue', 'purple', 'orange', 'cyan', 'black']

        for idx, target_far in enumerate(target_fars):
            ti = np.quantile(unknown_scores, target_far)
            calculated_thresholds[target_far] = ti

            false_accepts = np.sum(unknown_scores <= ti)
            actual_far = false_accepts / len(unknown_scores)

            true_accepts = np.sum(known_scores <= ti)
            dir_rate = true_accepts / len(known_scores) if len(known_scores) > 0 else 0
            frr_rate = 1.0 - dir_rate

            print(f"Target FAR: {target_far * 100}% -> Soglia Ti: {ti:.4f}")
            print(f"  > Valid. Actual FAR: {actual_far * 100:.2f}%")
            print(f"  > Valid. Est. DIR:   {dir_rate * 100:.2f}%")
            print(f"  > Valid. Est. FRR:   {frr_rate * 100:.2f}%")
            print("-" * 30)

            c = colors[idx % len(colors)]
            plt.axvline(ti, color=c, linestyle='--', linewidth=2, label=f'FAR {target_far * 100}% (Ti={ti:.2f})')

        plt.xlabel('Best Matching Distance')
        plt.ylabel('Density')
        plt.title('Validation Distances & Thresholds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return calculated_thresholds

    def evaluate_open_set_test_multiple(self, test_samples, test_labels, thresholds_dict):
        if self.gallery_embeddings_tensor is None:
            self._load_gallery_to_gpu()

        print(f"\n--- VALUTAZIONE TEST SET MULTIPLA (Open-Set) ---")
        print(f"Numero thresholds da testare: {len(thresholds_dict)}")

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

        best_indices = np.argmin(dists_np, axis=1)
        min_dists = dists_np[np.arange(len(dists_np)), best_indices]
        pred_ids = gal_lbl_np[best_indices]

        min_dists_unknown = np.full(total_attempts, float('inf'))
        for i, true_label in enumerate(test_lbl_np):
            mask_others = (gal_lbl_np != true_label)
            if np.any(mask_others):
                min_dists_unknown[i] = np.min(dists_np[i, mask_others])

        for target_far, ti in thresholds_dict.items():
            correct_id = 0
            false_reject = 0
            misidentified = 0
            unknown_accepted = 0

            for i, true_label in enumerate(test_lbl_np):
                dist_known = min_dists[i]
                pred_id = pred_ids[i]

                if dist_known > ti:
                    false_reject += 1
                elif pred_id == true_label:
                    correct_id += 1
                else:
                    misidentified += 1

                if min_dists_unknown[i] != float('inf'):
                    if min_dists_unknown[i] <= ti:
                        unknown_accepted += 1

            dir_metric = correct_id / total_attempts
            frr_metric = (false_reject + misidentified) / total_attempts
            far_metric = unknown_accepted / total_attempts
            tnr_metric = 1 - far_metric

            print(f"Target FAR: {target_far * 100}% | Soglia Ti: {ti:.4f}")
            print(f"  [Known]   DIR: {dir_metric * 100:.2f}% | FRR: {frr_metric * 100:.2f}%")
            print(f"  [Unknown] TNR: {tnr_metric * 100:.2f}% | FAR: {far_metric * 100:.2f}%")
            print("-" * 50)

    def evaluate_rank_n_performance(self, test_samples, test_labels, ti, k_ranks=None):

        if k_ranks is None:
            k_ranks = [1, 2]
        if self.gallery_embeddings_tensor is None:
            self._load_gallery_to_gpu()

        print(f"\n--- VALUTAZIONE RANK-N (CMC) & OPEN SET (Soglia Ti={ti:.4f}) ---")

        combined_data = list(zip(test_samples, test_labels))
        test_dataset = ListDataset(combined_data, transform=self.single_transform)
        loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

        test_embeddings = []
        inference_times = []

        print(f"Estrazione feature su {len(test_samples)} campioni...")

        self.resnet.eval()

        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self.device)

                start_t = time.time()

                emb = self.resnet(imgs)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

                end_t = time.time()

                batch_time = end_t - start_t
                inference_times.append(batch_time / len(imgs))

                test_embeddings.append(emb)

        if not test_embeddings:
            print("ERRORE: Nessun dato di test processato.")
            return

        avg_inference_time = sum(inference_times) / len(inference_times)

        test_emb_tensor = torch.cat(test_embeddings)

        dists = torch.cdist(test_emb_tensor, self.gallery_embeddings_tensor, p=2)

        dists_np = dists.cpu().numpy()
        test_lbl_np = np.array(test_labels)
        gal_lbl_np = self.gallery_labels

        total_probes = len(test_lbl_np)

        rank_counts = {k: 0 for k in k_ranks}

        dir_rank1 = 0
        frr_rank1 = 0

        false_accepts_impostor = 0
        total_impostor_attempts = 0

        for i, true_label in enumerate(test_lbl_np):

            sorted_indices = np.argsort(dists_np[i])

            for k in k_ranks:
                top_k_indices = sorted_indices[:k]
                top_k_labels = gal_lbl_np[top_k_indices]

                if true_label in top_k_labels:
                    rank_counts[k] += 1

            best_idx = sorted_indices[0]
            best_dist = dists_np[i, best_idx]
            pred_id = gal_lbl_np[best_idx]

            if pred_id == true_label and best_dist <= ti:
                dir_rank1 += 1
            else:
                frr_rank1 += 1

            mask_others = (gal_lbl_np != true_label)

            if np.any(mask_others):
                min_dist_impostor = np.min(dists_np[i, mask_others])
                total_impostor_attempts += 1

                if min_dist_impostor <= ti:
                    false_accepts_impostor += 1

        far_rank1 = (false_accepts_impostor / total_impostor_attempts) if total_impostor_attempts > 0 else 0

        print("-" * 50)
        print(f"Tempo Medio Inferenza (per immagine): {avg_inference_time:.6f} sec")
        print("-" * 50)

        print(">>> CLOSED SET IDENTIFICATION (CMC):")
        for k in sorted(k_ranks):
            accuracy = rank_counts[k] / total_probes
            print(f"    Rank-{k}: {accuracy * 100:.2f}%")

        print("-" * 50)
        print(f">>> OPEN SET PERFORMANCE (al Rank-1 con Ti={ti:.4f}):")
        print(f"    DIR (Correct ID & Accept): {(dir_rank1 / total_probes) * 100:.2f}%")
        print(f"    FRR (False Reject/MisID):  {(frr_rank1 / total_probes) * 100:.2f}%")
        print(f"    FAR (False Accept Impostor): {far_rank1 * 100:.2f}%")
        print("-" * 50)

    def evaluate_spoof_far_liveness_off(self, spoof_dir, ti):
        """
        Calcola il Rank-1 FAR sui Replay Attacks con Liveness DISATTIVATO.
        Simula lo scenario in cui il sistema è vulnerabile e accetta foto/video.
        """
        if self.gallery_embeddings_tensor is None:
            self._load_gallery_to_gpu()

        print(f"\n--- VALUTAZIONE REPLAY ATTACK (Liveness OFF) ---")
        print(f"Soglia Identificazione (Ti): {ti:.4f}")

        spoof_samples = []
        if isinstance(spoof_dir, list):
            spoof_samples = spoof_dir
        elif os.path.exists(spoof_dir):
            for root, _, files in os.walk(spoof_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        spoof_samples.append(os.path.join(root, f))

        if not spoof_samples:
            print("Nessun sample di spoof trovato.")
            return 0.0

        print(f"Estrazione features da {len(spoof_samples)} immagini spoof...")

        dummy_labels = [0] * len(spoof_samples)
        data_pairs = list(zip(spoof_samples, dummy_labels))

        spoof_dataset = ListDataset(data_pairs, transform=self.single_transform)
        loader = DataLoader(spoof_dataset, batch_size=32, shuffle=False, num_workers=2)

        spoof_embeddings = []
        self.resnet.eval()

        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                emb = self.resnet(imgs)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                spoof_embeddings.append(emb)

        if not spoof_embeddings:
            return 0.0

        spoof_emb_tensor = torch.cat(spoof_embeddings)

        dists = torch.cdist(spoof_emb_tensor, self.gallery_embeddings_tensor, p=2)
        dists_np = dists.cpu().numpy()

        total_attempts = len(spoof_samples)
        false_accepts = 0

        for i in range(total_attempts):
            min_dist = np.min(dists_np[i])

            if min_dist <= ti:
                false_accepts += 1

        spoof_far = false_accepts / total_attempts if total_attempts > 0 else 0

        print("-" * 40)
        print(f"Totale Attacchi Replay: {total_attempts}")
        print(f"Attacchi Accettati (False Accepts): {false_accepts}")
        print(f"Rank-1 FAR (Replay, Liveness OFF): {spoof_far * 100:.2f}%")
        print("-" * 40)

        return spoof_far

    def evaluate_spoof_far_liveness_on(self, spoof_dir, liveness_model, tf, ti, liveness_transform=None):
        """
        Calcola il Rank-1 FAR sui Replay Attacks con Liveness ATTIVO.
        Simula il sistema completo: Liveness Check -> (se passa) -> Identification.

        Args:
            spoof_dir: Path o lista di file spoof.
            liveness_model: Il modello SpoofingModel già caricato (su GPU).
            tf: Soglia Liveness (Threshold Fake).
            ti: Soglia Identificazione (Threshold Identification).
            liveness_transform: Trasformazione per l'immagine liveness (es. resize 160x160).
                                Se None, usa quella standard.
        """
        import os
        from PIL import Image
        from torchvision import transforms

        if self.gallery_embeddings_tensor is None:
            self._load_gallery_to_gpu()

        if liveness_transform is None:
            liveness_transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        print(f"\n--- VALUTAZIONE REPLAY ATTACK (Liveness ON) ---")
        print(f"Soglia Liveness (Tf): {tf} | Soglia ID (Ti): {ti:.4f}")

        spoof_samples = []
        if isinstance(spoof_dir, list):
            spoof_samples = spoof_dir
        elif os.path.exists(spoof_dir):
            for root, _, files in os.walk(spoof_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        spoof_samples.append(os.path.join(root, f))

        total_attempts = len(spoof_samples)
        if total_attempts == 0:
            print("Nessun sample trovato.")
            return None

        self.resnet.eval()
        liveness_model.eval()

        blocked_by_liveness = 0
        passed_liveness = 0
        system_false_accepts = 0

        print(f"Processando {total_attempts} attacchi...")

        with torch.no_grad():
            for fpath in spoof_samples:
                try:
                    img_pil = Image.open(fpath).convert('RGB')

                    img_live = liveness_transform(img_pil).unsqueeze(0).to(self.device)

                    logits = liveness_model(img_live)
                    liveness_score = torch.sigmoid(logits).item()

                    is_classified_live = (liveness_score > tf)

                    if not is_classified_live:
                        blocked_by_liveness += 1
                        continue

                    passed_liveness += 1

                    img_id = self.single_transform(img_pil).unsqueeze(0).to(self.device)

                    emb = self.resnet(img_id)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)

                    dists = torch.cdist(emb, self.gallery_embeddings_tensor)
                    min_dist = torch.min(dists).item()

                    if min_dist <= ti:
                        system_false_accepts += 1

                except Exception as e:
                    print(f"Errore su {fpath}: {e}")
                    continue

        liveness_far = passed_liveness / total_attempts if total_attempts > 0 else 0
        system_far = system_false_accepts / total_attempts if total_attempts > 0 else 0

        print("-" * 50)
        print(f"Totale Attacchi: {total_attempts}")
        print(f"Bloccati dal Liveness: {blocked_by_liveness} ({(blocked_by_liveness / total_attempts) * 100:.2f}%)")
        print(f"Passati dal Liveness:  {passed_liveness} (Liveness Failure Rate: {liveness_far * 100:.2f}%)")
        print(f"Accettati dal Sistema: {system_false_accepts} (Attacchi Riusciti)")
        print("-" * 50)
        print(f">>> SYSTEM RANK-1 FAR (Liveness ON): {system_far * 100:.2f}%")
        print("-" * 50)
        return system_far
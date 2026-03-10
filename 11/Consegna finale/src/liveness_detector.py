import os
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN


class LivenessDatasetBuilder:
    def __init__(self, metadata_df, source_video_dir, output_dataset_dir):
        """
        Classe unificata per la creazione del dataset di Liveness Detection.
        Include bilanciamento automatico delle classi Real vs Spoof.
        """
        self.metadata = metadata_df.copy()
        self.source_root = source_video_dir
        self.output_dir = output_dataset_dir
        self.splits = ['train', 'val', 'test']

        if 'Label' in self.metadata.columns and 'LiveOrSpoof' not in self.metadata.columns:
            self.metadata['LiveOrSpoof'] = self.metadata['Label'].apply(
                lambda x: 'live' if 'real' in str(x).lower() else 'spoof'
            )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"MTCNN running on device: {self.device}")

        self.mtcnn = MTCNN(
            device=self.device,
            select_largest=True,
            margin=20,
            post_process=False
        )

    def _get_real_video_path(self, kaggle_path):
        flattened_name = kaggle_path.replace("/", "-")
        full_path = os.path.join(self.source_root, flattened_name)

        if not os.path.exists(full_path):
            base_name_no_ext = os.path.splitext(flattened_name)[0]
            alternatives = ['.MOV', '.mp4', '.avi', '.mkv']
            for ext in alternatives:
                attempt_path = os.path.join(self.source_root, base_name_no_ext + ext)
                if os.path.exists(attempt_path):
                    return attempt_path
        return full_path

    def create_splits(self, val_size=0.2, test_size=0.1, random_state=42):
        """
        Divide i dati basandosi sui VIDEO univoci MANTENENDO LA PROPORZIONE (Stratify)
        tra le classi Live e Spoof.
        """
        print("Calcolo degli split Train/Val/Test sui video univoci con STRATIFY...")

        unique_videos = self.metadata['Video_Path'].unique()

        video_labels = []
        for vid in unique_videos:
            lbl = self.metadata[self.metadata['Video_Path'] == vid]['LiveOrSpoof'].iloc[0]
            video_labels.append(lbl)

        train_videos, temp_videos, train_labels, temp_labels = train_test_split(
            unique_videos,
            video_labels,
            test_size=(val_size + test_size),
            random_state=random_state,
            stratify=video_labels
        )

        relative_test_size = test_size / (val_size + test_size)

        val_videos, test_videos = train_test_split(
            temp_videos,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=temp_labels
        )

        split_map = {}
        for v in train_videos: split_map[v] = 'train'
        for v in val_videos: split_map[v] = 'val'
        for v in test_videos: split_map[v] = 'test'

        self.metadata['split'] = self.metadata['Video_Path'].map(split_map)

        print(f"Split completato (Stratified).")
        print(f"Video Train: {len(train_videos)} | Val: {len(val_videos)} | Test: {len(test_videos)}")

        train_lbls = [video_labels[i] for i, v in enumerate(unique_videos) if v in train_videos]
        print(f"Proporzione Live in Train (Video): {train_lbls.count('live') / len(train_lbls):.2%}")

        return self.metadata

    def verify_split_integrity(self):
        """
        Verifica che non ci siano intersezioni tra Train, Val e Test
        a livello di VIDEO.
        """

        print("\n--- VERIFICA INTEGRITÀ SPLIT ---")
        train_vids = set(self.metadata[self.metadata['split']=='train']['Video_Path'].unique())
        val_vids = set(self.metadata[self.metadata['split']=='val']['Video_Path'].unique())
        test_vids = set(self.metadata[self.metadata['split']=='test']['Video_Path'].unique())

        tv_leak = train_vids.intersection(val_vids)
        tt_leak = train_vids.intersection(test_vids)
        vt_leak = val_vids.intersection(test_vids)

        if len(tv_leak) == 0 and len(tt_leak) == 0 and len(vt_leak) == 0:
            print("SUCCESSO: Nessun video è condiviso tra gli split. Data Leakage impossibile.")
        else:
            print("ALLARME: Trovata intersezione tra gli split!")
            print(f"Video in Train & Val: {len(tv_leak)}")
            print(f"Video in Train & Test: {len(tt_leak)}")
            print(f"Video in Val & Test: {len(vt_leak)}")


    def extract_and_create(self, frames_per_video=15, crop_faces=True, target_size=(224, 224)):
        """
        Processa il dataset bilanciando le classi: estrae più frame dai video 'live'
        per compensare il minor numero di video rispetto agli attacchi.
        """

        if 'split' not in self.metadata.columns:
            raise ValueError("Errore: Esegui prima .create_splits()!")

        n_live = self.metadata[self.metadata['LiveOrSpoof'] == 'live'].shape[0]
        n_spoof = self.metadata[self.metadata['LiveOrSpoof'] == 'spoof'].shape[0]

        balance_multiplier = 1.0
        if 0 < n_live < n_spoof:
            balance_multiplier = n_spoof / n_live

        print(f"\n--- Configurazione Bilanciamento ---")
        print(f"Video Live: {n_live} | Video Spoof: {n_spoof}")
        print(f"Base frames per spoof: {frames_per_video}")
        print(f"Target frames per live: {int(frames_per_video * balance_multiplier)} (Moltiplicatore: {balance_multiplier:.2f}x)")
        print("-" * 30)

        for split in self.splits:
            for label in ['live', 'spoof']:
                os.makedirs(os.path.join(self.output_dir, split, label), exist_ok=True)

        print(f"Inizio estrazione dataset in: {self.output_dir}")
        counters = {'processed': 0, 'missing': 0, 'errors': 0}

        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            original_path = row['Video_Path']
            real_path = self._get_real_video_path(original_path)
            label = row['LiveOrSpoof']
            split = row['split']

            if not os.path.exists(real_path):
                counters['missing'] += 1
                continue

            cap = cv2.VideoCapture(real_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                cap.release()
                counters['errors'] += 1
                continue

            current_target_frames = frames_per_video
            if label == 'live':
                current_target_frames = int(frames_per_video * balance_multiplier)

            frame_indices = np.linspace(0, total_frames - 1, current_target_frames, dtype=int)

            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: continue

                final_img = frame

                if crop_faces:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)

                    try:
                        boxes, _ = self.mtcnn.detect(pil_img)
                        if boxes is not None:
                            box = boxes[0]
                            x1, y1, x2, y2 = [int(b) for b in box]
                            x1 = max(0, x1); y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

                            final_img = frame[y1:y2, x1:x2]
                            if final_img.size == 0: continue
                        else:
                            continue
                    except Exception as e:
                        continue

                try:
                    final_img = cv2.resize(final_img, target_size)
                    video_basename = os.path.basename(original_path).replace('/', '_').replace('.', '_')
                    filename = f"{video_basename}_f{i}.jpg"
                    save_path = os.path.join(self.output_dir, split, label, filename)
                    cv2.imwrite(save_path, final_img)
                except Exception as e:
                    counters['errors'] += 1
                    continue

            cap.release()
            counters['processed'] += 1

        print("\n--- Processo Completato ---")
        print(f"Video processati: {counters['processed']}")
        print(f"Video mancanti: {counters['missing']}")
        print(f"Errori lettura/resize: {counters['errors']}")



class SpoofingDataset(Dataset):
    def __init__(self, root_dir, split='train', target_size=(160, 160)):
        """
        Args:
            root_dir (str): Path della cartella principale (contiene train/val/test).
            split (str): 'train', 'val', o 'test'.
            target_size (tuple): Dimensione finale delle immagini (es. 160x160 per Facenet).
        """
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size

        self.class_to_idx = {'spoof': 0, 'live': 1}

        self.split_dir = os.path.join(root_dir, split)
        if not os.path.exists(self.split_dir):
            raise ValueError(f"La cartella {self.split_dir} non esiste!")

        self.live_samples = self._load_samples_from_folder('live', label=1)
        self.spoof_samples = self._load_samples_from_folder('spoof', label=0)

        print(f"[{split.upper()}] Originali -> Live: {len(self.live_samples)} | Spoof: {len(self.spoof_samples)}")

        self.samples = []
        if split == 'train' and len(self.live_samples) > 0 and len(self.spoof_samples) > 0:
            if len(self.live_samples) < len(self.spoof_samples):
                imbalance_ratio = len(self.spoof_samples) // len(self.live_samples)
                remainder = len(self.spoof_samples) % len(self.live_samples)

                print(f"   >>> Bilanciamento attivo: Moltiplico i dati Live x{imbalance_ratio}")

                self.samples.extend(self.live_samples * imbalance_ratio)
                self.samples.extend(self.live_samples[:remainder])
                self.samples.extend(self.spoof_samples)
            else:
                self.samples = self.live_samples + self.spoof_samples
        else:
            self.samples = self.live_samples + self.spoof_samples

        print(f"[{split.upper()}] Finali (dopo bilanciamento) -> Totale: {len(self.samples)}")

        self.base_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            # Normalizzazione standard (es. ImageNet o [-1, 1] per Facenet)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.live_augment = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_samples_from_folder(self, class_name, label):
        """Carica i path delle immagini da una sottocartella."""
        folder_path = os.path.join(self.split_dir, class_name)
        samples = []
        if os.path.exists(folder_path):
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.heic')):
                    samples.append((os.path.join(folder_path, fname), label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Errore caricamento {img_path}: {e}")
            return torch.zeros(3, *self.target_size), label

        if self.split == 'train' and label == 1:
            image_tensor = self.live_augment(image)
        else:
            image_tensor = self.base_transform(image)

        return image_tensor, label


class SpoofingModel(nn.Module):
    """
    Modello di rete neurale per il rilevamento di Face Spoofing (Liveness Detection).

    Questa classe implementa un classificatore binario basato sull'architettura ResNet50.
    Utilizza una strategia di "Partial Fine-Tuning": i primi strati della rete (che estraggono
    feature generiche come bordi e forme) vengono congelati, mentre l'ultimo blocco convoluzionale
    (layer4) e il classificatore finale vengono lasciati liberi di apprendere (trainable).
    """

    def __init__(self, pretrained=True):
        """
        Inizializza il modello SpoofingModel.

        Configura la backbone ResNet50, gestisce il freezing dei pesi
        e sostituisce l'ultimo layer per la classificazione binaria.
        """
        super(SpoofingModel, self).__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        for param in self.backbone.layer3.parameters():
            param.requires_grad = True

        for param in self.backbone.layer2.parameters():
            param.requires_grad = True

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Definisce il forward pass.
        """
        return self.backbone(x)
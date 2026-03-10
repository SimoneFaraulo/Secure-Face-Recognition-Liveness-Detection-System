import torch
import numpy as np
import re
from facenet_pytorch import InceptionResnetV1
from config import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class GalleryDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        all_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        def get_subject_id(name):
            match = re.search(r'\d+', name)
            return int(match.group()) if match else 999999

        subjects = sorted(all_folders, key=get_subject_id)

        for subject in subjects:
            label = get_subject_id(subject)
            subj_path = os.path.join(root_dir, subject)

            files = [f for f in os.listdir(subj_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            files = sorted(files)

            for file in files:
                self.samples.append((os.path.join(subj_path, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


class FaceRecognitionSystem:
    def __init__(self, device, weights_path=None):
        self.device = device
        print(f"Inizializzazione Modello su {self.device}...")

        self.resnet = InceptionResnetV1(pretrained=None, classify=False).to(self.device)

        if weights_path and os.path.exists(weights_path):
            print(f"--- Caricamento pesi custom da: {weights_path} ---")

            checkpoint = torch.load(weights_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            self.resnet.load_state_dict(state_dict, strict=False)

            print(">>> Pesi caricati con successo!")
        else:
            print("ATTENZIONE: Nessun peso custom trovato/fornito. Uso VGGFace2 standard.")
            self.resnet = InceptionResnetV1(pretrained='vggface2', classify=False).to(self.device)

        self.resnet.eval()

        self.gallery_embeddings = None
        self.gallery_labels = None

        self.single_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def build_and_save_gallery(self, gallery_dir, prefix_embeddings, prefix_labels):
        print(f"\n--- Costruzione Gallery da: {gallery_dir} ---")
        dataset = GalleryDataset(gallery_dir)

        if len(dataset) == 0:
            print("Nessuna immagine trovata nella gallery!")
            return

        loader = DataLoader(dataset, batch_size=BATCH_SIZE_GALLERY, shuffle=False, num_workers=2)

        emb_list = []
        lbl_list = []

        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(self.device)
                embeddings = self.resnet(imgs)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                emb_list.append(embeddings.cpu().numpy())
                lbl_list.append(lbls.numpy())

        self.gallery_embeddings = np.concatenate(emb_list)
        self.gallery_labels = np.concatenate(lbl_list)

        np.save(f"{prefix_embeddings}.npy", self.gallery_embeddings)
        np.save(f"{prefix_labels}.npy", self.gallery_labels)

        print(f"Gallery Salvata: {self.gallery_embeddings.shape[0]} volti.")
        print(f"Files: {prefix_embeddings}.npy, {prefix_labels}.npy")

        self._load_gallery_to_gpu()

    def load_existing_gallery(self, load_path_prefix_embeddings, load_path_prefix_labels):
        """Carica una gallery precedentemente salvata."""
        print(f"\nCaricamento Gallery da file: {load_path_prefix_labels} e {load_path_prefix_embeddings}...")
        try:
            self.gallery_embeddings = np.load(f"{load_path_prefix_embeddings}.npy")
            self.gallery_labels = np.load(f"{load_path_prefix_labels}.npy")
            self._load_gallery_to_gpu()
            print(f"Gallery caricata: {len(self.gallery_labels)} soggetti.")
        except FileNotFoundError:
            print("Errore: File gallery non trovati.")


    def _load_gallery_to_gpu(self):
        """Metodo interno per spostare la gallery su GPU una volta sola."""
        if self.gallery_embeddings is not None:
            self.gallery_embeddings_tensor = torch.tensor(self.gallery_embeddings).to(self.device)


    def predict_single(self, img_path):
        """
        Prende UN path, estrae features, confronta con tutta la gallery.
        """
        if self.gallery_embeddings is None:
            raise ValueError("La Gallery non è stata caricata! Esegui build o load prima.")

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            return None, float('inf')

        img_tensor = self.single_transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            probe_emb = self.resnet(img_tensor)
            probe_emb = torch.nn.functional.normalize(probe_emb, p=2, dim=1)

        dists = torch.cdist(probe_emb, self.gallery_embeddings_tensor, p=2)

        min_dist, min_idx = torch.min(dists, dim=1)

        match_idx = min_idx.item()
        distance = min_dist.item()

        predicted_label = self.gallery_labels[match_idx]

        return predicted_label, distance

    def print_memory(self):
        if self.device.type == 'cuda':
            alloc = torch.cuda.memory_allocated(self.device) / 1024 ** 2
            print(f"[Memory Check] GPU Allocata: {alloc:.2f} MB")
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix

class SpoofingEvaluator:
    def __init__(self, model, device, config):
        """
        Classe unica per gestire la valutazione, il calcolo delle soglie e i test finali.
        """
        self.model = model
        self.device = device
        self.config = config
        self.model.to(self.device)

    def load_checkpoint(self, filename):
        """Carica i pesi del modello da un file specifico gestendo errori e formati."""
        ckpt_path = os.path.join(self.config['checkpoint_dir'], filename)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Errore: Il file {ckpt_path} non esiste.")

        print(f"--- Caricamento pesi da: {filename} ---")
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                best_loss = checkpoint.get('best_loss', 'N/A')
                epoch_saved = checkpoint.get('epoch', 'N/A')
                print(f"Pesi caricati con successo. (Epoca: {epoch_saved}, Best Loss: {best_loss})")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Pesi caricati con successo.")

        except Exception as e:
            print(f"Errore nel caricamento dei pesi: {e}")
            raise e

    def _get_predictions(self, data_loader):
        """Metodo interno (helper) per ottenere score e label da un DataLoader."""
        self.model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.sigmoid(logits)

                all_scores.extend(probs.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())

        return np.array(all_scores), np.array(all_labels)

    def find_threshold_fixed_far(self, val_loader, target_far=0.01):
        """
        Trova la soglia che garantisce un FAR (False Acceptance Rate) massimo specificato.
        Esempio: target_far=0.01 significa che accettiamo al massimo l'1% di Spoof come veri.
        """
        print(f"\n--- 1. Ricerca Soglia per FAR <= {target_far:.1%} ---")

        scores, labels = self._get_predictions(val_loader)

        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        frr = 1 - tpr

        valid_indices = np.where(fpr <= target_far)[0]

        if len(valid_indices) == 0:
            print("Attenzione: Impossibile raggiungere questo FAR con il modello attuale (tutti gli spoof passano).")
            best_idx = 0
        else:
            best_idx = valid_indices[-1]

        fixed_threshold = thresholds[best_idx]
        actual_far = fpr[best_idx]
        resulting_frr = frr[best_idx]

        print(f"   >>> Soglia Trovata:       {fixed_threshold:.4f}")
        print(f"   >>> FAR Effettivo:        {actual_far:.2%} (Target: {target_far:.1%})")
        print(f"   >>> FRR Risultante:       {resulting_frr:.2%} (Live rifiutati)")

        self._plot_roc(fpr, tpr, best_idx, actual_far, target_far)

        return fixed_threshold

    def find_threshold_eer(self, val_loader):
        """
        Trova la soglia di Equal Error Rate (EER) dove FAR ≈ FRR.
        È il miglior compromesso generale se non hai requisiti di sicurezza specifici.
        """
        print("\n--- 2. Ricerca della Soglia di Equilibrio (EER) ---")

        scores, labels = self._get_predictions(val_loader)

        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        frr = 1 - tpr

        best_index = np.nanargmin(np.abs(fpr - frr))

        best_threshold = thresholds[best_index]
        best_eer = fpr[best_index]
        actual_frr = frr[best_index]

        print(f"   >>> Soglia EER:           {best_threshold:.4f}")
        print(f"   >>> EER (Errore Medio):   {best_eer:.2%}")
        print(f"   >>> Dettaglio: FAR {fpr[best_index]:.2%} | FRR {actual_frr:.2%}")

        self._plot_eer(thresholds, fpr, frr, best_threshold, best_eer)

        return best_threshold

    def analyze_errors(self, data_loader, threshold):
        """
        Analizza e stampa i file specifici che vengono sbagliati dal modello.
        IMPORTANTE: Richiede che il DataLoader abbia shuffle=False per mappare correttamente i file.
        """
        print(f"\n--- 3. Analisi Dettagliata Errori (Soglia: {threshold:.4f}) ---")

        self.model.eval()
        try:
            dataset_samples = data_loader.dataset.samples
        except AttributeError:
            print("Impossibile recuperare i percorsi dei file dal dataset. Salto l'analisi dettagliata.")
            return

        fp_list = []
        fn_list = []

        global_idx = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.sigmoid(logits).view(-1)

                preds = (probs >= threshold).long().cpu().numpy()
                probs_np = probs.cpu().numpy()
                labels_np = labels.cpu().numpy()

                for i in range(len(labels_np)):
                    pred = preds[i]
                    label = labels_np[i]
                    score = float(probs_np[i])

                    if global_idx < len(dataset_samples):
                        img_path = dataset_samples[global_idx][0]
                    else:
                        img_path = "Unknown"

                    if label == 0 and pred == 1:
                        fp_list.append((img_path, score))
                    elif label == 1 and pred == 0:
                        fn_list.append((img_path, score))

                    global_idx += 1

        print(f"Falsi Positivi (Spoof passati): {len(fp_list)}")
        for path, score in fp_list:
            print(f"   Score: {score:.4f} | {os.path.basename(path)}")

        print(f"Falsi Negativi (Live bloccati): {len(fn_list)}")
        for path, score in fn_list:
            print(f"   Score: {score:.4f} | {os.path.basename(path)}")


    def evaluate_test_set(self, test_loader, threshold):
        """
        Esegue la valutazione finale sul Test Set usando la soglia scelta.
        Stampa le metriche biometriche standard (FAR, FRR, HTER).
        """
        print(f"\n--- 4. REPORT FINALE TEST SET (Soglia: {threshold:.4f}) ---")

        scores, labels = self._get_predictions(test_loader)
        preds = (scores >= threshold).astype(int)

        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()

        far = fp / (tn + fp) if (tn + fp) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        hter = (far + frr) / 2
        acc = accuracy_score(labels, preds)

        print(f"  METRICHE:")
        print(f"   - FAR (Sicurezza):  {far:.2%}")
        print(f"   - FRR (Usabilità):  {frr:.2%}")
        print(f"   - HTER:             {hter:.2%}")
        print(f"   - ACCURACY:         {acc:.2%}")

        print(f"  DETTAGLIO:")
        print(f"   - Spoof Bloccati (TN): {tn}")
        print(f"   - Spoof Passati  (FP): {fp} (ATTACCHI RIUSCITI)")
        print(f"   - Live Passati   (TP): {tp}")
        print(f"   - Live Bloccati  (FN): {fn} (UTENTI BLOCCATI)")

        self._plot_confusion_matrix(cm, threshold)

    def _plot_roc(self, fpr, tpr, best_idx, actual_far, target_far):
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f'Chosen (FAR={actual_far:.2%})')
        plt.axvline(x=target_far, color='green', linestyle='--', label='Max FAR limit')
        plt.xlabel('False Acceptance Rate (FAR)')
        plt.ylabel('True Acceptance Rate (TAR)')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def _plot_eer(self, thresholds, fpr, frr, best_thresh, best_eer):
        plt.figure(figsize=(6, 4))
        plt.plot(thresholds, fpr, label='FAR (Spoof Accettati)', color='red')
        plt.plot(thresholds, frr, label='FRR (Live Rifiutati)', color='blue')
        plt.axvline(x=best_thresh, color='green', linestyle='--', label=f'Thresh: {best_thresh:.2f}')
        plt.scatter(best_thresh, best_eer, color='black', zorder=5)
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')
        plt.title(f'EER Point ({best_eer:.2%})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def _plot_confusion_matrix(self, cm, threshold):
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Spoof (Pred)', 'Live (Pred)'],
                    yticklabels=['Spoof (Real)', 'Live (Real)'])
        plt.title(f'Confusion Matrix @ {threshold:.4f}')
        plt.ylabel('Reale')
        plt.xlabel('Predetto')
        plt.show()

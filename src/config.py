import os

#DATASET UNPACKING
DATASET_ROOT = os.getenv("DATASET_ROOT", '/content/drive/MyDrive/BBA_Dataset')
DATASET_PROCESSED_ROOT = os.getenv("DATASET_PROCESSED_ROOT", '/content/drive/MyDrive/BBA_Dataset_processed')
DATASET_ANTISPOOF_ROOT = os.getenv("DATASET_ANTISPOOF_ROOT", '/content/drive/MyDrive/Spoof')


#FEATURE EXTRACTOR
BATCH_SIZE_GALLERY = int(os.getenv('BATCH_SIZE_GALLERY', '32'))
BASE_DIR = os.getenv('BASE_DIR', './drive/MyDrive/BBA_Dataset_processed')
GALLERY_DIR = os.getenv('GALLERY_DIR', os.path.join(BASE_DIR, 'galleries'))
PROBES_DIR = os.getenv('PROBES_DIR', os.path.join(BASE_DIR, 'probes'))

#LIVENESS DETECTOR
SOURCE_ROOT = os.getenv('SOURCE_ROOT', "/content/drive/MyDrive/anti_spoofing_dataset/dataset")
CSV_PATH = os.getenv('CSV_PATH', "/content/drive/MyDrive/anti_spoofing_dataset/dataset/metadata.csv")
VIDEOS_DIR = os.getenv('VIDEOS_DIR', "/content/drive/MyDrive/anti_spoofing_dataset/dataset/videos")
DATASET_ANTISPOOF_FINAL = os.getenv('DATASET_ANTISPOOF_FINAL', "/content/drive/MyDrive/dataset_v1")
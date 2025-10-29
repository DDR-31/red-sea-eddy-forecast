import numpy as np
import xarray as xr
import os
import tensorflow as tf

# Impor dari file-file kita sebelumnya
from data_generator import DataGenerator
from build_model import build_unet_convlstm_model, N_TIME_STEPS

# --- Konfigurasi Training ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_dataset_with_bathy.nc')
STATS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'normalization_stats.json')
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Parameter yang bisa kamu 'tuning' (sesuaikan)
BATCH_SIZE = 2    # Turunkan jika VRAM tidak cukup, naikkan jika bisa
EPOCHS = 10       # Mulai dengan sedikit epoch (10-20)
VAL_SPLIT = 0.2   # 20% data untuk validasi

print(">>> Memulai proses training model...")

# --- 1. Persiapan Indeks Data ---
print("    Mempersiapkan indeks train/validation split...")
# Hitung total indeks yang mungkin
total_possible_samples = len(xr.open_dataset(DATA_PATH)['time']) - N_TIME_STEPS
all_indices = np.arange(total_possible_samples)

# Acak indeksnya agar data latih & validasi terdistribusi
np.random.shuffle(all_indices)

split_point = int(len(all_indices) * (1.0 - VAL_SPLIT))
train_indices = all_indices[:split_point]
val_indices = all_indices[split_point:]

print(f"    Total sampel: {total_possible_samples}")
print(f"    Sampel latih: {len(train_indices)}")
print(f"    Sampel validasi: {len(val_indices)}")

# --- 2. Inisialisasi Data Generators ---
print("    Menginisialisasi Data Generators...")
train_gen = DataGenerator(
    data_path=DATA_PATH,
    stats_path=STATS_PATH,
    batch_size=BATCH_SIZE,
    n_time_steps=N_TIME_STEPS,
    indices=train_indices
)

val_gen = DataGenerator(
    data_path=DATA_PATH,
    stats_path=STATS_PATH,
    batch_size=BATCH_SIZE,
    n_time_steps=N_TIME_STEPS,
    indices=val_indices
)

# --- 3. Membangun Model ---
print("    Membangun arsitektur model...")
model = build_unet_convlstm_model()
model.summary()

# --- 4. Callbacks (Pembantu Training) ---
# ModelCheckpoint: Menyimpan model HANYA jika performa di data validasi membaik
checkpoint_path = os.path.join(MODEL_SAVE_DIR, "convlstm_best.keras")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,       # Hanya simpan yang terbaik
    monitor='val_loss',        # Pantau 'loss' di data validasi
    mode='min',
    verbose=1
)

# EarlyStopping: Hentikan training jika 'val_loss' tidak membaik
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3, # Hentikan jika tidak ada kemajuan selama 3 epoch
    mode='min',
    verbose=1,
    restore_best_weights=True # Kembalikan ke bobot terbaik saat berhenti
)

# --- 5. Mulai Training! ---
print("\n" + "="*50)
print(f"    MEMULAI TRAINING (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE})")
print("="*50 + "\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[model_checkpoint, early_stopping],
    verbose=1 # Tampilkan progress bar
)

print("\n" + "="*50)
print("    TRAINING SELESAI")
print(f"    Model terbaik disimpan di: {checkpoint_path}")
print("="*50 + "\n")

import numpy as np
import xarray as xr
import json
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm # Library untuk progress bar, install: pip install tqdm

# Impor dari file-file kita
from data_generator import DataGenerator
from build_model import N_TIME_STEPS, HEIGHT, WIDTH

# --- Konfigurasi Path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_dataset.nc')
STATS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'normalization_stats.json')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'convlstm_best.keras')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

print(">>> Memulai kalkulasi Peta RMSE Spasial...")

# --- 1. Muat Model & Statistik ---
print("    Memuat model dan statistik...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(STATS_PATH, 'r') as f:
    stats = json.load(f)

sla_mean = stats['sla']['mean']
sla_std = stats['sla']['std']

# --- 2. Siapkan Data Generator Validasi (SELURUHNYA) ---
total_possible_samples = len(xr.open_dataset(DATA_PATH)['time']) - N_TIME_STEPS
all_indices = np.arange(total_possible_samples)
split_point = int(len(all_indices) * 0.8)
val_indices = all_indices[split_point:]

print(f"    Mempersiapkan generator untuk {len(val_indices)} sampel validasi...")
# PENTING: Batch size 1 untuk memproses satu per satu
val_gen = DataGenerator(
    data_path=DATA_PATH,
    stats_path=STATS_PATH,
    batch_size=1, 
    n_time_steps=N_TIME_STEPS,
    indices=val_indices
)

# --- 3. Iterasi & Kumpulkan Semua Prediksi ---
# Kita akan kumpulkan semua (Error)^2 di sini
all_squared_errors = []

print("    Melakukan prediksi pada seluruh data validasi (ini akan memakan waktu)...")
# tqdm akan memberi kita progress bar yang bagus
for i in tqdm(range(len(val_gen))):
    X_val, y_val_normalized = val_gen[i]
    
    # Lakukan prediksi
    y_pred_normalized = model.predict(X_val, verbose=0) # verbose=0 agar tidak spam
    
    # Denormalisasi
    y_val_actual = (y_val_normalized * sla_std) + sla_mean
    y_pred_actual = (y_pred_normalized * sla_std) + sla_mean
    
    # Hitung Squared Error (Error Kuadrat)
    squared_error = np.square(y_val_actual - y_pred_actual)
    
    all_squared_errors.append(squared_error[0]) # [0] untuk hapus dimensi batch

# --- 4. Hitung Peta RMSE ---
print("    Menghitung peta RMSE...")
# Tumpuk semua error menjadi satu array besar
all_squared_errors_np = np.stack(all_squared_errors, axis=0)

# Hitung rata-rata error di sepanjang dimensi waktu
mean_squared_error = np.nanmean(all_squared_errors_np, axis=0)

# Ambil akar kuadrat untuk mendapatkan RMSE
rmse_map = np.sqrt(mean_squared_error)

# --- 5. Visualisasi Peta RMSE ---
print("    Membuat plot...")
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(rmse_map[:, :, 0], cmap='inferno') # cmap 'inferno' bagus untuk error
ax.set_title('Peta RMSE Spasial (Error dalam meter)')
ax.invert_yaxis()
fig.colorbar(im, ax=ax, shrink=0.8, label='RMSE (meter)')

plt.tight_layout()
output_plot_path = os.path.join(REPORT_DIR, 'spatial_rmse_map.png')
plt.savefig(output_plot_path)

print(f"\n✅ Sukses! Peta RMSE disimpan di:")
print(f"   {output_plot_path}")
print("   Buka file ini. Area yang 'terang' (kuning/putih) adalah di mana modelmu paling sering salah.")

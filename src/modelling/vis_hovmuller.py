import numpy as np
import xarray as xr
import json
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm 

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

# --- KONFIGURASI PLOT ---
LAT_SLICE_INDEX = 108 

print(">>> Memulai kalkulasi Diagram Hovmöller...")

# --- 1. Muat Model & Statistik ---
print("    Memuat model dan statistik...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(STATS_PATH, 'r') as f:
    stats = json.load(f)

sla_mean = stats['sla']['mean']
sla_std = stats['sla']['std']

# --- 2. Siapkan Data Generator Validasi (SELURUHNYA, TIDAK ACAK) ---
total_possible_samples = len(xr.open_dataset(DATA_PATH)['time']) - N_TIME_STEPS
all_indices = np.arange(total_possible_samples)
split_point = int(len(all_indices) * 0.8)
val_indices = all_indices[split_point:] # 20% terakhir, berurutan

print(f"    Mempersiapkan generator untuk {len(val_indices)} sampel validasi...")
val_gen = DataGenerator(
    data_path=DATA_PATH,
    stats_path=STATS_PATH,
    batch_size=1, 
    n_time_steps=N_TIME_STEPS,
    indices=val_indices
)

# --- 3. Iterasi & Kumpulkan Irisan (Slices) ---
all_actual_slices = []
all_pred_slices = []

print(f"    Memprediksi dan mengambil irisan di lintang {LAT_SLICE_INDEX}...")
for i in tqdm(range(len(val_gen))):
    X_val, y_val_normalized = val_gen[i]
    y_pred_normalized = model.predict(X_val, verbose=0)
    
    y_val_actual = (y_val_normalized * sla_std) + sla_mean
    y_pred_actual = (y_pred_normalized * sla_std) + sla_mean
    
    slice_actual = y_val_actual[0, LAT_SLICE_INDEX, :, 0] 
    slice_pred = y_pred_actual[0, LAT_SLICE_INDEX, :, 0]
    
    all_actual_slices.append(slice_actual)
    all_pred_slices.append(slice_pred)

# --- 4. Tumpuk Irisan Menjadi Gambar Hovmöller ---
print("    Menyusun gambar Hovmöller...")
actual_hovmoller = np.stack(all_actual_slices, axis=0)
pred_hovmoller = np.stack(all_pred_slices, axis=0)

# --- 5. Visualisasi (REVISI v4) ---
print("    Membuat plot (Metode Anti-Gagal)...")
fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=True) # <-- Buat lebih lebar
fig.suptitle(f"Diagram Hovmöller (SLA @ Latitude index {LAT_SLICE_INDEX})", fontsize=16)

vmax = np.nanmax(actual_hovmoller)
vmin = np.nanmin(actual_hovmoller)
if vmin == vmax: vmax += 1e-6 

# --- Plot 1: Data Asli ---
ax = axes[0]
im1 = ax.imshow(actual_hovmoller, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
ax.set_title("Data Asli (Ground Truth)")
ax.set_ylabel("Waktu (Hari dalam set validasi)")
ax.set_xlabel("Indeks Longitude (Barat ke Timur)")
# 💡 REVISI: Buat colorbar HANYA untuk axis ini
fig.colorbar(im1, ax=ax, shrink=0.7, label='SLA (meter)')

# --- Plot 2: Prediksi Model ---
ax = axes[1]
im2 = ax.imshow(pred_hovmoller, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
ax.set_title("Prediksi Model")
ax.set_xlabel("Indeks Longitude (Barat ke Timur)")
# 💡 REVISI: Buat colorbar HANYA untuk axis ini
fig.colorbar(im2, ax=ax, shrink=0.7, label='SLA (meter)')

# Gunakan tight_layout standar
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
output_plot_path = os.path.join(REPORT_DIR, 'hovmoller_diagram.png')
plt.savefig(output_plot_path)

print(f"\n✅ Sukses! Diagram Hovmöller (REVISI v4) disimpan di:")
print(f"   {output_plot_path}")
print("   Plot ini akan memiliki dua colorbar, tapi seharusnya sudah benar posisinya.")

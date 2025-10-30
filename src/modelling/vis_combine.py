import numpy as np
import xarray as xr
import json
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as gridspec # For complex layouts

# Impor dari file-file kita
from data_generator import DataGenerator
from build_model import N_TIME_STEPS, HEIGHT, WIDTH # N_CHANNELS will be 5 now

print(">>> Memulai Visualisasi Gabungan (Hasil, RMSE, Hovmöller)...")

# --- Konfigurasi ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_dataset_with_bathy.nc') # <-- File baru
STATS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'normalization_stats.json') # <-- File baru (5 var)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'convlstm_best.keras') # <-- Model baru
REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

LAT_SLICE_INDEX = 108 # Untuk Hovmöller
EXAMPLE_INDEX = 500   # Indeks sampel (dari set validasi) untuk contoh prediksi

# --- 1. Muat Model & Statistik ---
print("    Memuat model dan statistik...")
model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
with open(STATS_PATH, 'r') as f:
    stats = json.load(f)

sla_mean = stats['sla']['mean']
sla_std = stats['sla']['std']

# --- 2. Siapkan Indeks Validasi ---
total_possible_samples = len(xr.open_dataset(DATA_PATH)['time']) - N_TIME_STEPS
all_indices = np.arange(total_possible_samples)
split_point = int(len(all_indices) * 0.8)
val_indices = all_indices[split_point:] # 20% terakhir, berurutan

print(f"    Total sampel validasi: {len(val_indices)}")

# --- 3. Kalkulasi untuk Plot ---
# Kita perlu iterasi sekali untuk mendapatkan semua prediksi & data asli
all_actual_val = []
all_pred_val = []
all_squared_errors = []
all_actual_slices = []
all_pred_slices = []

print("    Memproses seluruh data validasi (Prediksi, RMSE, Hovmöller)...")
# Buat generator dengan batch_size=1
val_gen = DataGenerator(
    data_path=DATA_PATH,
    stats_path=STATS_PATH,
    batch_size=1,
    n_time_steps=N_TIME_STEPS,
    indices=val_indices
)

for i in tqdm(range(len(val_gen))):
    X_val, y_val_normalized = val_gen[i]
    y_pred_normalized = model.predict(X_val, verbose=0)

    # Denormalisasi
    y_val_actual = (y_val_normalized * sla_std) + sla_mean
    y_pred_actual = (y_pred_normalized * sla_std) + sla_mean

    # Kumpulkan data untuk RMSE
    squared_error = np.square(y_val_actual - y_pred_actual)
    all_squared_errors.append(squared_error[0]) # Hapus dimensi batch

    # Kumpulkan data untuk Hovmöller
    slice_actual = y_val_actual[0, LAT_SLICE_INDEX, :, 0]
    slice_pred = y_pred_actual[0, LAT_SLICE_INDEX, :, 0]
    all_actual_slices.append(slice_actual)
    all_pred_slices.append(slice_pred)

    # Simpan juga data asli & prediksi (jika diperlukan untuk plot contoh)
    all_actual_val.append(y_val_actual[0])
    all_pred_val.append(y_pred_actual[0])


# --- 4. Hitung Peta RMSE & Data Hovmöller ---
print("    Menghitung peta RMSE dan data Hovmöller...")
# RMSE
all_squared_errors_np = np.stack(all_squared_errors, axis=0)
mean_squared_error = np.nanmean(all_squared_errors_np, axis=0)
rmse_map = np.sqrt(mean_squared_error)

# Hovmöller
actual_hovmoller = np.stack(all_actual_slices, axis=0)
pred_hovmoller = np.stack(all_pred_slices, axis=0)

# Ambil data untuk plot contoh
example_actual = all_actual_val[EXAMPLE_INDEX]
example_pred = all_pred_val[EXAMPLE_INDEX]
example_error = example_actual - example_pred


# --- 5. Membuat Plot Gabungan (Versi Profesional) ---
print("    Membuat plot gabungan...")
fig = plt.figure(figsize=(18, 13)) # Sedikit lebih tinggi untuk judul
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3) # Tambah spasi

# Ambil koordinat untuk label (opsional tapi bagus)
ds_coord = xr.open_dataset(DATA_PATH)
lat_coord = ds_coord['latitude'].values
lon_coord = ds_coord['longitude'].values
time_coord_val = ds_coord['time'].isel(time=val_indices).values # Waktu validasi

# --- Baris 1: Contoh Prediksi ---
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# Tentukan rentang dan tick untuk colorbar contoh
vmax_ex = np.nanmax(np.abs(example_actual)) # Max absolute value
vmin_ex = -vmax_ex
levels_ex = np.linspace(vmin_ex, vmax_ex, 11) # 11 level warna

# Plot 1a: Ground Truth
im1 = ax1.imshow(example_actual[:,:,0], cmap='RdBu_r', vmin=vmin_ex, vmax=vmax_ex,
                 extent=[lon_coord.min(), lon_coord.max(), lat_coord.min(), lat_coord.max()])
ax1.set_title(f"a) Ground Truth SLA (m)\n(Validation Day {EXAMPLE_INDEX})", fontsize=12)
ax1.set_xlabel("Longitude (°E)")
ax1.set_ylabel("Latitude (°N)")
fig.colorbar(im1, ax=ax1, shrink=0.7, label='SLA (m)', ticks=levels_ex, format='%.2f')

# Plot 1b: Prediksi
im2 = ax2.imshow(example_pred[:,:,0], cmap='RdBu_r', vmin=vmin_ex, vmax=vmax_ex,
                 extent=[lon_coord.min(), lon_coord.max(), lat_coord.min(), lat_coord.max()])
ax2.set_title(f"b) Predicted SLA (m)\n(Validation Day {EXAMPLE_INDEX})", fontsize=12)
ax2.set_xlabel("Longitude (°E)")
ax2.set_ylabel("Latitude (°N)")
fig.colorbar(im2, ax=ax2, shrink=0.7, label='SLA (m)', ticks=levels_ex, format='%.2f')

# Plot 1c: Error
vmax_err_ex = np.nanmax(np.abs(example_error))
if vmax_err_ex == 0: vmax_err_ex = 1e-6
levels_err = np.linspace(-vmax_err_ex, vmax_err_ex, 11)
im3 = ax3.imshow(example_error[:,:,0], cmap='coolwarm', vmin=-vmax_err_ex, vmax=vmax_err_ex,
                 extent=[lon_coord.min(), lon_coord.max(), lat_coord.min(), lat_coord.max()])
ax3.set_title(f"c) Prediction Error (m)\n(Truth - Prediction)", fontsize=12)
ax3.set_xlabel("Longitude (°E)")
ax3.set_ylabel("Latitude (°N)")
fig.colorbar(im3, ax=ax3, shrink=0.7, label='Error (m)', ticks=levels_err, format='%.2f')

# --- Baris 2 Kiri: Peta RMSE ---
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.imshow(rmse_map[:, :, 0], cmap='inferno',
                 extent=[lon_coord.min(), lon_coord.max(), lat_coord.min(), lat_coord.max()])
ax4.set_title('d) Spatial Root Mean Square Error', fontsize=12)
ax4.set_xlabel("Longitude (°E)")
ax4.set_ylabel("Latitude (°N)")
fig.colorbar(im4, ax=ax4, shrink=0.7, label='RMSE (m)')

# --- Baris 2 Tengah & Kanan: Hovmöller ---
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2], sharey=ax5) # Share Y axis

# Tentukan rentang dan tick untuk colorbar Hovmöller
vmax_hov = np.nanmax(np.abs(actual_hovmoller))
vmin_hov = -vmax_hov
levels_hov = np.linspace(vmin_hov, vmax_hov, 11)

# Ambil tanggal untuk label sumbu Y Hovmoller (opsional)
start_date_val = np.datetime_as_string(time_coord_val[0], unit='D')
end_date_val = np.datetime_as_string(time_coord_val[-1], unit='D')


im5 = ax5.imshow(actual_hovmoller, cmap='RdBu_r', vmin=vmin_hov, vmax=vmax_hov, aspect='auto',
                 extent=[lon_coord.min(), lon_coord.max(), len(val_indices), 0]) # Y-axis reversed (time goes down)
ax5.set_title(f"e) Hovmöller: Ground Truth SLA (m)\n(@ Lat {lat_coord[LAT_SLICE_INDEX]:.2f}°N)", fontsize=12)
ax5.set_ylabel(f"Days in Validation Period\n({start_date_val} to {end_date_val})")
ax5.set_xlabel("Longitude (°E)")

im6 = ax6.imshow(pred_hovmoller, cmap='RdBu_r', vmin=vmin_hov, vmax=vmax_hov, aspect='auto',
                 extent=[lon_coord.min(), lon_coord.max(), len(val_indices), 0])
ax6.set_title(f"f) Hovmöller: Predicted SLA (m)\n(@ Lat {lat_coord[LAT_SLICE_INDEX]:.2f}°N)", fontsize=12)
ax6.set_xlabel("Longitude (°E)")
# Matikan label Y di plot kedua agar tidak tumpang tindih
plt.setp(ax6.get_yticklabels(), visible=False)


# Colorbar tunggal untuk Hovmöller
cbar_ax_hov = fig.add_axes([0.93, 0.1, 0.015, 0.35]) # Posisi manual [kiri, bawah, lebar, tinggi]
fig.colorbar(im6, cax=cbar_ax_hov, label='SLA (m)', ticks=levels_hov, format='%.2f')

# --- Finalisasi ---
fig.suptitle("U-Net ConvLSTM Model Performance Summary: Red Sea SLA Prediction", fontsize=16, y=0.99)
# Menggunakan tight_layout dengan penyesuaian agar colorbar Hovmoller tidak tertimpa
plt.tight_layout(rect=[0, 0.03, 0.91, 0.96])

output_plot_path = os.path.join(REPORT_DIR, 'combined_visualization_professional.png') # Nama file baru
plt.savefig(output_plot_path, dpi=200) # Tingkatkan DPI untuk kualitas lebih baik

print(f"\n✅ Sukses! Plot gabungan (profesional) disimpan di:")
print(f"   {output_plot_path}")

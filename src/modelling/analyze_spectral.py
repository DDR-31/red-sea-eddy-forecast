import numpy as np
import xarray as xr
import json
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal # For detrending

# Impor dari file-file kita
from data_generator import DataGenerator
from build_model import N_TIME_STEPS, HEIGHT, WIDTH # N_CHANNELS needed for generator

print(">>> Memulai Analisis Spektral...")

# --- Konfigurasi ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_dataset_with_bathy.nc')
STATS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'normalization_stats.json')
# 💡 Pastikan nama file ini benar!
MODEL_ORIGINAL_PATH = os.path.join(PROJECT_ROOT, 'models', 'convlstm_best.keras')
MODEL_UNET_PATH = os.path.join(PROJECT_ROOT, 'models', 'convlstm_best.keras')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

# Grid spacing (approximate, needed for wavenumber calculation)
# Roughly 0.083 degrees ~ 9.2 km at the equator. Let's use km.
DX = 9.2 # km

# --- Fungsi Bantuan untuk Spektrum ---
def calculate_radial_psd(data_map, dx):
    """
    Menghitung Power Spectral Density (PSD) 1D yang dirata-ratakan secara radial
    dari peta 2D.
    """
    # 0. Pastikan tidak ada NaN (ganti daratan dengan rata-rata laut?)
    #    Untuk simpelnya, ganti NaN dengan 0 (karena sudah denormalisasi)
    data_map_filled = np.nan_to_num(data_map, nan=0.0)

    # 1. Detrend: Hilangkan tren linear (penting untuk FFT)
    data_detrended = signal.detrend(data_map_filled, axis=0)
    data_detrended = signal.detrend(data_detrended, axis=1)

    # 2. Apply window (optional tapi bagus): Hanning window
    ny, nx = data_detrended.shape
    window = np.outer(np.hanning(ny), np.hanning(nx))
    data_windowed = data_detrended * window

    # 3. Hitung 2D FFT
    fft_result = np.fft.fftshift(np.fft.fft2(data_windowed))

    # 4. Hitung Power Spectrum 2D (magnitude squared)
    psd_2d = np.abs(fft_result)**2 / (nx * ny)**2 # Normalisasi

    # 5. Hitung wavenumber radial
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) # cycles per km
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    wavenumber_radial = np.sqrt(kx_grid**2 + ky_grid**2)

    # 6. Binning Radial
    # Tentukan bin wavenumber (misal dari 0 hingga Nyquist)
    k_max = np.max(wavenumber_radial)
    k_bins = np.linspace(0, k_max, min(nx, ny) // 2) # Jumlah bin = setengah resolusi
    k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    # Rata-ratakan PSD di setiap cincin wavenumber
    psd_1d = np.zeros(len(k_bin_centers))
    for i in range(len(k_bin_centers)):
        mask = (wavenumber_radial >= k_bins[i]) & (wavenumber_radial < k_bins[i+1])
        if np.any(mask):
            psd_1d[i] = psd_2d[mask].mean()
        else:
            psd_1d[i] = np.nan # Atau 0

    return k_bin_centers, psd_1d


# --- 1. Muat Model & Statistik ---
print("    Memuat model dan statistik...")
# Muat model pertama (ConvLSTM + Bathy)
try:
    model_original = tf.keras.models.load_model(MODEL_ORIGINAL_PATH, safe_mode=False)
except OSError:
    print(f"❌ Error: Tidak dapat memuat model asli di {MODEL_ORIGINAL_PATH}", file=sys.stderr)
    print("    Pastikan file model ada dan namanya benar.", file=sys.stderr)
    sys.exit(1)

# Muat model kedua (U-Net + Bathy)
try:
    model_unet = tf.keras.models.load_model(MODEL_UNET_PATH, safe_mode=False)
except OSError:
     print(f"❌ Error: Tidak dapat memuat model U-Net di {MODEL_UNET_PATH}", file=sys.stderr)
     print("    Pastikan file model ada dan namanya benar.", file=sys.stderr)
     sys.exit(1)


with open(STATS_PATH, 'r') as f:
    stats = json.load(f)
sla_mean = stats['sla']['mean']
sla_std = stats['sla']['std']

# --- 2. Siapkan Data Generator Validasi ---
total_possible_samples = len(xr.open_dataset(DATA_PATH)['time']) - N_TIME_STEPS
all_indices = np.arange(total_possible_samples)
split_point = int(len(all_indices) * 0.8)
val_indices = all_indices[split_point:]

print(f"    Mempersiapkan generator untuk {len(val_indices)} sampel validasi...")
val_gen = DataGenerator(
    data_path=DATA_PATH,
    stats_path=STATS_PATH,
    batch_size=1,
    n_time_steps=N_TIME_STEPS,
    indices=val_indices
)

# --- 3. Iterasi, Prediksi & Hitung Spektrum ---
all_spectra_actual = []
all_spectra_original = []
all_spectra_unet = []

print("    Memproses data validasi & menghitung spektrum...")
for i in tqdm(range(len(val_gen))):
    X_val, y_val_normalized = val_gen[i]

    # Prediksi dari kedua model
    y_pred_norm_orig = model_original.predict(X_val, verbose=0)
    y_pred_norm_unet = model_unet.predict(X_val, verbose=0)

    # Denormalisasi
    y_val_actual = (y_val_normalized * sla_std) + sla_mean
    y_pred_actual_orig = (y_pred_norm_orig * sla_std) + sla_mean
    y_pred_actual_unet = (y_pred_norm_unet * sla_std) + sla_mean

    # Ambil peta 2D (hapus dimensi batch & channel)
    map_actual = y_val_actual[0, :, :, 0]
    map_pred_orig = y_pred_actual_orig[0, :, :, 0]
    map_pred_unet = y_pred_actual_unet[0, :, :, 0]

    # Hitung spektrum untuk masing-masing
    k, spec_actual = calculate_radial_psd(map_actual, DX)
    _, spec_orig = calculate_radial_psd(map_pred_orig, DX)
    _, spec_unet = calculate_radial_psd(map_pred_unet, DX)

    # Simpan spektrum (jika valid)
    if not np.isnan(spec_actual).all(): # Cek jika spektrum valid
        all_spectra_actual.append(spec_actual)
        all_spectra_original.append(spec_orig)
        all_spectra_unet.append(spec_unet)

# --- 4. Rata-ratakan Spektrum ---
print("    Merata-ratakan spektrum...")
if not all_spectra_actual:
    print("❌ Error: Tidak ada spektrum valid yang dihitung!", file=sys.stderr)
    sys.exit(1)

mean_spec_actual = np.nanmean(np.stack(all_spectra_actual, axis=0), axis=0)
mean_spec_original = np.nanmean(np.stack(all_spectra_original, axis=0), axis=0)
mean_spec_unet = np.nanmean(np.stack(all_spectra_unet, axis=0), axis=0)

# Ambil wavenumber dari perhitungan terakhir (semua sama)
wavenumbers = k

# --- 5. Visualisasi Spektrum ---
print("    Membuat plot spektrum...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(wavenumbers, mean_spec_actual, label='Data Asli (Ground Truth)', color='k', linewidth=2)
ax.loglog(wavenumbers, mean_spec_original, label='Model Asli (ConvLSTM+Bathy)', color='blue', linestyle='--')
ax.loglog(wavenumbers, mean_spec_unet, label='Model U-Net (+Bathy)', color='red', linestyle=':')

ax.set_xlabel("Wavenumber (siklus / km)")
ax.set_ylabel("Power Spectral Density (m^2 / (siklus/km))")
ax.set_title("Perbandingan Spektrum Daya SLA Rata-rata")
ax.grid(True, which="both", ls="-", alpha=0.5)
ax.legend()

plt.tight_layout()
output_plot_path = os.path.join(REPORT_DIR, 'spectral_analysis.png')
plt.savefig(output_plot_path)

print(f"\n✅ Sukses! Plot analisis spektral disimpan di:")
print(f"   {output_plot_path}")

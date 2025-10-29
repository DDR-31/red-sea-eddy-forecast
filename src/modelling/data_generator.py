import numpy as np
import xarray as xr
import json
import os
import tensorflow as tf

# Impor konfigurasi model kita dari skrip sebelumnya
from build_model import N_TIME_STEPS, HEIGHT, WIDTH, N_CHANNELS

class DataGenerator(tf.keras.utils.Sequence):
    """
    Versi 2.0: Sekarang menerima daftar 'indices' untuk
    membedakan antara data train dan validation.
    """
    def __init__(self, data_path, stats_path, batch_size, n_time_steps, indices):
        print(f"Menginisialisasi DataGenerator...")
        
        self.ds = xr.open_dataset(data_path)
        print(f"    Dataset dibuka dari: {data_path}")
        
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
        print(f"    Statistik normalisasi dimuat dari: {stats_path}")

        self.batch_size = batch_size
        self.n_time_steps = n_time_steps
        self.variables = ['sla', 'thetao', 'u10', 'v10', 'bathy']
        
        # 💡 PERUBAHAN UTAMA: Gunakan indices yang diberikan
        self.indices = indices
        self.total_samples = len(self.indices) # Total sampel sekarang berdasarkan indices
        
        print(f"    Generator ini akan mengelola {self.total_samples} sampel.")

    def __len__(self):
        """Memberi tahu Keras ada berapa batch dalam satu epoch"""
        # Kita tidak pakai 'floor' agar semua sampel terpakai
        return int(np.ceil(self.total_samples / self.batch_size))

    def __getitem__(self, index):
        """Mengambil satu batch data"""

        # 💡 --- FIX: Calculate indices FIRST ---
        start_idx = index * self.batch_size
        # Pastikan tidak 'overflow' di batch terakhir
        end_idx = min((index + 1) * self.batch_size, self.total_samples)

        # Dapatkan indeks sampel untuk batch ini dari daftar
        batch_indices = self.indices[start_idx:end_idx]
        # ------------------------------------

        # Ukuran batch bisa lebih kecil di iterasi terakhir
        current_batch_size = len(batch_indices)

        # Siapkan array kosong
        X = np.empty((current_batch_size, self.n_time_steps, HEIGHT, WIDTH, N_CHANNELS))
        y = np.empty((current_batch_size, HEIGHT, WIDTH, 1))

        # --- 💡 Muat Batimetri SATU KALI di awal batch ---
        bathy_data = self.ds['bathy'].values # Muat peta bathy (H, W)
        bathy_mean = self.stats['bathy']['mean']
        bathy_std = self.stats['bathy']['std']
        if bathy_std == 0: bathy_std = 1 # Hindari pembagian dengan nol
        bathy_normalized = (bathy_data - bathy_mean) / bathy_std
        bathy_normalized_filled = np.nan_to_num(bathy_normalized, nan=0.0)
        # Tambahkan dimensi channel
        bathy_input_slice = np.expand_dims(bathy_normalized_filled, axis=-1) # Bentuk jadi (H, W, 1)
        # ----------------------------------------------------

        for i, sample_idx in enumerate(batch_indices):

            # --- Persiapan Input (X) ---
            input_start_time = sample_idx
            input_end_time = sample_idx + self.n_time_steps

            # 💡 Hanya muat 4 variabel pertama yang punya waktu
            vars_with_time = self.variables[:-1] # Ambil sla, thetao, u10, v10
            x_data = self.ds[vars_with_time].isel(time=slice(input_start_time, input_end_time))

            # Loop untuk 4 channel pertama
            for c_idx, var in enumerate(vars_with_time):
                mean = self.stats[var]['mean']
                std = self.stats[var]['std']
                normalized_data = (x_data[var].values - mean) / std
                X[i, :, :, :, c_idx] = np.nan_to_num(normalized_data, nan=0.0)

            # 💡 Masukkan data batimetri yang sudah dinormalisasi ke channel terakhir
            # Kita 'tile' (ulangi) peta bathy sebanyak N_TIME_STEPS
            # bathy_input_slice bentuknya (H, W, 1), kita ulangi di axis=0 (waktu)
            X[i, :, :, :, -1:] = np.tile(bathy_input_slice, (self.n_time_steps, 1, 1, 1))

            # --- Persiapan Target (y) ---
            target_time_idx = input_end_time
            y_data = self.ds['sla'].isel(time=target_time_idx).values
            sla_mean = self.stats['sla']['mean']
            sla_std = self.stats['sla']['std']
            y_normalized = (y_data - sla_mean) / sla_std
            y_normalized_filled = np.nan_to_num(y_normalized, nan=0.0)
            y[i] = np.expand_dims(y_normalized_filled, axis=-1)

        return X, y

if __name__ == "__main__":
    # 💡 PERUBAHAN TES: Kita simulasikan pemisahan data
    print("\n>>> Melakukan tes DataGenerator (v2.0)...")
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_dataset_with_bathy.nc')
    STATS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'normalization_stats.json')
    
    BATCH_SIZE = 4
    
    # Hitung total indeks yang mungkin
    total_possible_samples = len(xr.open_dataset(DATA_PATH)['time']) - N_TIME_STEPS
    all_indices = np.arange(total_possible_samples)
    
    # Simulasikan: 80% train, 20% val
    split_point = int(len(all_indices) * 0.8)
    train_indices = all_indices[:split_point]
    val_indices = all_indices[split_point:]
    
    print(f"    Total indeks: {len(all_indices)}")
    print(f"    Indeks Latih: {len(train_indices)}")
    print(f"    Indeks Validasi: {len(val_indices)}")

    # Tes generator Latih
    print("\n    Tes Generator Latih...")
    train_gen = DataGenerator(DATA_PATH, STATS_PATH, BATCH_SIZE, N_TIME_STEPS, train_indices)
    X_batch, y_batch = train_gen[0] # Ambil batch pertama
    print(f"    Bentuk X Latih: {X_batch.shape}")
    print(f"    Bentuk y Latih: {y_batch.shape}")

    # Tes generator Validasi
    print("\n    Tes Generator Validasi...")
    val_gen = DataGenerator(DATA_PATH, STATS_PATH, BATCH_SIZE, N_TIME_STEPS, val_indices)
    X_batch, y_batch = val_gen[0] # Ambil batch pertama
    print(f"    Bentuk X Validasi: {X_batch.shape}")
    print(f"    Bentuk y Validasi: {y_batch.shape}")
    
    print(f"\n✅ Tes v2.0 berhasil.")

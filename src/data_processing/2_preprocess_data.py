import xarray as xr
import os
import sys

# --- Pathing ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'cmems_data.nc')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'processed_data.nc')

os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

# --- Proses Pra-pemrosesan ---
print(">>> Memulai skrip pra-pemrosesan data...")

# 1. Muat Dataset Mentah
try:
    print(f"    Memuat data mentah dari: {RAW_DATA_PATH}")
    # Gunakan 'chunks' agar xarray tidak memuat semua ke RAM sekaligus
    ds_raw = xr.open_dataset(RAW_DATA_PATH, chunks={'time': 100})
except FileNotFoundError:
    print(f"❌ Error: File data mentah tidak ditemukan di {RAW_DATA_PATH}", file=sys.stderr)
    print("    Pastikan kamu sudah menjalankan skrip 01_download_cmems.py terlebih dahulu.", file=sys.stderr)
    sys.exit(1)

# 💡 --- PERBAIKAN UTAMA DI SINI --- 💡
# Kita pilih HANYA lapisan permukaan (indeks kedalaman ke-0)
# Ini akan mengurangi penggunaan memori sebesar 98%
try:
    print("    Memilih lapisan permukaan (depth=0)...")
    ds_surface = ds_raw.isel(depth=0)
except ValueError:
    # Penjaga jika data ternyata sudah 2D (tidak ada 'depth')
    print("    Data sudah 2D (tidak ada dimensi 'depth'). Melanjutkan...")
    ds_surface = ds_raw
except Exception as e:
    print(f"❌ Error saat memilih 'depth': {e}", file=sys.stderr)
    print("    Cek nama dimensi kedalaman di file NetCDF-mu.", file=sys.stderr)
    sys.exit(1)

# 2. Hitung Rata-rata Klimatologis (Mean Sea Surface)
print("    Menghitung rata-rata klimatologis untuk 'zos'...")
# Gunakan .compute() untuk memaksa perhitungan yang efisien
zos_mean = ds_surface['zos'].mean(dim='time').compute()

# 3. Hitung Anomali Tinggi Muka Laut (sla)
print("    Menghitung 'sla' dari 'zos'...")
sla = ds_surface['zos'] - zos_mean
sla.attrs['long_name'] = 'Sea Level Anomaly'
sla.attrs['units'] = 'm'
sla.attrs['comment'] = 'Calculated as zos minus the 2007-2021 time-mean zos.'

# 4. Buat Dataset yang Sudah Diproses
print("    Membuat dataset yang sudah diproses dengan variabel 'sla' dan 'thetao'.")
ds_processed = xr.Dataset({
    'sla': sla,
    'thetao': ds_surface['thetao']
})

# 5. Penanganan Nilai yang Hilang (Missing Values)
print("    Menangani nilai yang hilang (jika ada)...")
# Sekarang operasi ini hanya berjalan pada data 2D+time, jauh lebih ringan!
ds_processed = ds_processed.ffill(dim='time').bfill(dim='time')

# 6. Simpan Dataset yang Sudah Diproses
print(f"    Menyimpan data yang sudah diproses ke: {PROCESSED_DATA_PATH}")
# Gunakan .compute() untuk mengeksekusi semua operasi "malas" dan menyimpannya
ds_processed.compute().to_netcdf(PROCESSED_DATA_PATH)

print("\n✅ Sukses! Pra-pemrosesan data selesai.")
print(f"    Dataset baru siap digunakan di: {PROCESSED_DATA_PATH}")

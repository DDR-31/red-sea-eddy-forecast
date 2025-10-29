'''
import xarray as xr
import os
import sys
import glob

# --- Pathing ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

OCEAN_DATA_PATH = os.path.join(PROCESSED_DIR, 'processed_data.nc')
ERA5_FILES_PATTERN = os.path.join(RAW_DIR, 'era5_data_*.nc')
FINAL_DATA_PATH = os.path.join(PROCESSED_DIR, 'final_dataset.nc')

print(">>> Memulai skrip penggabungan dan regridding data (VERSI 2.0)...")

try:
    # 1. Muat data Oseanografi (Target kita)
    print(f"    Memuat data laut dari: {OCEAN_DATA_PATH}")
    ds_ocean = xr.open_dataset(OCEAN_DATA_PATH)
except FileNotFoundError:
    print(f"❌ Error: File {OCEAN_DATA_PATH} tidak ditemukan!", file=sys.stderr)
    sys.exit(1)

# 2. Muat dan Gabungkan semua file ERA5
print(f"    Mencari file ERA5 di: {ERA5_FILES_PATTERN}")
era5_files = sorted(glob.glob(ERA5_FILES_PATTERN))
if not era5_files:
    print("❌ Error: Tidak ada file data ERA5 yang ditemukan!", file=sys.stderr)
    sys.exit(1)

print(f"    Menemukan {len(era5_files)} file ERA5. Menggabungkan...")
ds_era5 = xr.open_mfdataset(era5_files)

# 3. 💡 PERBAIKAN: Ganti nama koordinat 'valid_time' menjadi 'time'
if 'valid_time' in ds_era5.coords:
    print("    Mengganti nama 'valid_time' -> 'time' pada data ERA5.")
    ds_era5 = ds_era5.rename({'valid_time': 'time'})

# 4. 💡 PERBAIKAN: Lakukan Penjajaran Waktu (Time Alignment)
# Kita 'reindex' data ERA5 agar waktunya sama persis dengan ds_ocean.
# Ini akan otomatis memotong data ERA5 (Juli-Des 2021) yang tidak kita perlukan.
print(f"    Menyamakan rentang waktu ke data laut (total {len(ds_ocean['time'])} hari)...")
ds_era5_aligned = ds_era5.reindex(time=ds_ocean['time'], method='nearest')

# 5. Regridding (Interpolasi)
print("    Melakukan regridding data ERA5 agar sesuai dengan grid CMEMS...")
print("    Ini mungkin memakan waktu beberapa menit...")
ds_era5_regridded = ds_era5_aligned.interp_like(ds_ocean, method='linear')

# 6. Penanganan Nilai Hilang (NaN) pasca-interpolasi
print("    Menangani nilai NaN pasca-regridding...")
ds_era5_regridded = ds_era5_regridded.ffill(dim='latitude').bfill(dim='latitude')
ds_era5_regridded = ds_era5_regridded.ffill(dim='longitude').bfill(dim='longitude')

# 7. Gabungkan kedua dataset
print("    Menggabungkan data laut dan data atmosfer...")
# Sekarang 'time' sudah sejajar, merge akan berjalan mulus
ds_final = xr.merge([ds_ocean, ds_era5_regridded])

# 8. Simpan Dataset Final
print(f"    Menyimpan dataset final ke: {FINAL_DATA_PATH}")
ds_final.to_netcdf(FINAL_DATA_PATH)

print("\n✅ Sukses! Dataset final siap untuk machine learning.")
print(f"   Lokasi: {FINAL_DATA_PATH}")

print("\n--- Ringkasan Dataset Final (Seharusnya Sudah Sinkron) ---")
print(ds_final)
'''
import xarray as xr
import os
import sys
import glob

# --- Pathing ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

OCEAN_DATA_PATH = os.path.join(PROCESSED_DIR, 'processed_data.nc')
ERA5_FILES_PATTERN = os.path.join(RAW_DIR, 'era5_data_*.nc')
# 💡 Point back to the ORIGINAL GEBCO file
GEBCO_FILENAME = 'gebco_red_sea.nc' # <-- Make sure this is correct
GEBCO_PATH = os.path.join(RAW_DIR, GEBCO_FILENAME)
FINAL_DATA_PATH = os.path.join(PROCESSED_DIR, 'final_dataset_with_bathy.nc')

print(">>> Memulai skrip penggabungan data (VERSI 3.2 - Regrid Batimetri Di Sini)...")

try:
    # 1. Muat data Oseanografi (Target Grid)
    print(f"    Memuat data laut dari: {OCEAN_DATA_PATH}")
    ds_ocean = xr.open_dataset(OCEAN_DATA_PATH)
except FileNotFoundError:
    print(f"❌ Error: File {OCEAN_DATA_PATH} tidak ditemukan!", file=sys.stderr)
    sys.exit(1)

# 2. Muat dan Gabungkan ERA5
# ... (kode ERA5 tetap sama) ...
print(f"    Mencari dan menggabungkan file ERA5...")
era5_files = sorted(glob.glob(ERA5_FILES_PATTERN))
if not era5_files:
    print("❌ Error: Tidak ada file data ERA5!", file=sys.stderr)
    sys.exit(1)
ds_era5 = xr.open_mfdataset(era5_files)


# --- 💡 REVISI UTAMA: Muat GEBCO Asli & Regrid DI SINI ---
try:
    print(f"    Memuat data GEBCO asli dari: {GEBCO_PATH}")
    ds_gebco_raw = xr.open_dataset(GEBCO_PATH)
    if 'elevation' not in ds_gebco_raw:
        print("❌ Error: Variabel 'elevation' tidak ditemukan di file GEBCO.", file=sys.stderr)
        sys.exit(1)
    bathy_raw = ds_gebco_raw['elevation']

    # Lakukan regridding SEKARANG, menggunakan ds_ocean sebagai referensi
    print("    Melakukan regridding batimetri ke grid CMEMS...")
    # Penting: Rename dulu koordinat GEBCO jika namanya beda (misal, lat/lon vs latitude/longitude)
    # Sesuaikan 'lat' dan 'lon' jika nama koordinat di file GEBCO-mu berbeda
    bathy_renamed_coords = bathy_raw.rename({'lat': 'latitude', 'lon': 'longitude'}) 
    bathy_regridded = bathy_renamed_coords.interp_like(ds_ocean, method='linear')

    # Ubah daratan (positif) menjadi NaN dulu
    bathy_regridded = bathy_regridded.where(bathy_regridded <= 0)
    
    # Isi NaN (daratan) dengan 0
    print("    Mengisi nilai NaN (daratan) di batimetri dengan 0...")
    bathy_filled = bathy_regridded.fillna(0.0)
    # Ganti nama variabel jadi 'bathy'
    bathy_filled.name = 'bathy'

except FileNotFoundError:
    print(f"❌ Error: File GEBCO {GEBCO_PATH} tidak ditemukan!", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"❌ Error saat memproses GEBCO: {e}", file=sys.stderr)
    sys.exit(1)
# --- Akhir Revisi Blok Batimetri ---


# 3. Sinkronisasi Waktu ERA5
# ... (kode sinkronisasi ERA5 tetap sama) ...
if 'valid_time' in ds_era5.coords:
    ds_era5 = ds_era5.rename({'valid_time': 'time'})
ds_era5_aligned = ds_era5.reindex(time=ds_ocean['time'], method='nearest')

# 4. Regridding ERA5
# ... (kode regridding ERA5 tetap sama) ...
print("    Melakukan regridding data ERA5...")
ds_era5_regridded = ds_era5_aligned.interp_like(ds_ocean, method='linear')

# 5. Penanganan NaN ERA5
# ... (kode penanganan NaN ERA5 tetap sama) ...
print("    Menangani nilai NaN pasca-regridding ERA5...")
ds_era5_regridded = ds_era5_regridded.ffill(dim='latitude').bfill(dim='latitude')
ds_era5_regridded = ds_era5_regridded.ffill(dim='longitude').bfill(dim='longitude')


# 6. Gabungkan SEMUA dataset
print("    Menggabungkan data laut, atmosfer, dan batimetri...")
ds_final = xr.merge([ds_ocean, ds_era5_regridded, bathy_filled]) # Gunakan bathy_filled

# 7. Simpan Dataset Final
# ... (kode simpan tetap sama) ...
print(f"    Menyimpan dataset final ke: {FINAL_DATA_PATH}")
ds_final.to_netcdf(FINAL_DATA_PATH)

print("\n✅ Sukses! Dataset final dengan batimetri (v3.2) siap digunakan.")
print(f"   Lokasi: {FINAL_DATA_PATH}")

print("\n--- Ringkasan Dataset Final (Semoga Sinkron Sekarang) ---")
print(ds_final)

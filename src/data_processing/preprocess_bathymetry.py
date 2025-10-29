import xarray as xr
import os
import sys

# --- Pathing ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# GANTI INI dengan nama file GEBCO yang kamu unduh
GEBCO_FILENAME = 'gebco_red_sea.nc' 
GEBCO_PATH = os.path.join(RAW_DIR, GEBCO_FILENAME)
# File acuan grid (data laut kita yang sudah diproses)
GRID_REF_PATH = os.path.join(PROCESSED_DIR, 'processed_data.nc') 
OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'bathymetry_regridded.nc')

print(">>> Memulai pra-pemrosesan data batimetri...")

# --- 1. Muat Data Batimetri (GEBCO) ---
try:
    print(f"    Memuat data GEBCO dari: {GEBCO_PATH}")
    ds_gebco = xr.open_dataset(GEBCO_PATH)
    # Variabel utama di GEBCO biasanya bernama 'elevation'
    if 'elevation' not in ds_gebco:
        print("❌ Error: Variabel 'elevation' tidak ditemukan di file GEBCO.", file=sys.stderr)
        # Coba cek nama variabel lain jika bukan 'elevation'
        print("    Nama variabel yang ada:", list(ds_gebco.data_vars), file=sys.stderr)
        sys.exit(1)
    bathy = ds_gebco['elevation']
except FileNotFoundError:
    print(f"❌ Error: File GEBCO tidak ditemukan di {GEBCO_PATH}", file=sys.stderr)
    print("    Pastikan nama file sudah benar dan file ada di data/raw/", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"❌ Error saat memuat GEBCO: {e}", file=sys.stderr)
    sys.exit(1)

# --- 2. Muat Data Acuan Grid ---
try:
    print(f"    Memuat data acuan grid dari: {GRID_REF_PATH}")
    ds_ref = xr.open_dataset(GRID_REF_PATH)
except FileNotFoundError:
    print(f"❌ Error: File acuan grid {GRID_REF_PATH} tidak ditemukan!", file=sys.stderr)
    sys.exit(1)

# --- 3. Regridding (Interpolasi) ---
# Kita buat grid GEBCO mengikuti grid CMEMS (data laut kita)
print("    Melakukan regridding batimetri ke grid CMEMS...")
bathy_regridded = bathy.interp_like(ds_ref, method='linear')

# --- 4. Ubah Nilai Positif (Daratan) menjadi NaN ---
# Data GEBCO: negatif = kedalaman laut, positif = ketinggian darat
# Kita hanya peduli laut, jadi daratan kita buat NaN agar nanti bisa diisi 0
print("    Mengubah elevasi daratan (positif) menjadi NaN...")
bathy_regridded = bathy_regridded.where(bathy_regridded <= 0)

# --- 5. Konversi ke Kedalaman Positif (Opsional tapi Umum) ---
# Seringkali lebih intuitif bekerja dengan kedalaman positif
# bathy_regridded = -bathy_regridded 
# bathy_regridded.attrs['long_name'] = 'Bathymetry (Positive Depth)'
# bathy_regridded.attrs['units'] = 'm'
# Jika kamu uncomment ini, ingatlah saat normalisasi nanti.
# Untuk sekarang, kita biarkan negatif.

# --- 6. Simpan Hasil ---
print(f"    Menyimpan batimetri yang sudah diregrid ke: {OUTPUT_PATH}")
bathy_regridded.to_netcdf(OUTPUT_PATH)

print("\n✅ Sukses! Data batimetri siap digabungkan.")

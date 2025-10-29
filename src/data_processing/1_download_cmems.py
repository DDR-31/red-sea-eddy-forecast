import copernicusmarine as cm
import os
import sys

# --- Konfigurasi ---
# GANTI DENGAN KREDENSIAL CMEMS KAMU
# Sebaiknya gunakan environment variables untuk keamanan, tapi untuk sekarang kita hardcode dulu.
USERNAME_CMEMS = 'dsantosa'
PASSWORD_CMEMS = '31Desember!'

# --- Pathing yang Robust ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_DIRECTORY = os.path.join(PROJECT_ROOT, 'data', 'raw')
OUTPUT_FILENAME = 'cmems_data.nc'
FULL_OUTPUT_PATH = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)

# Pastikan direktori output ada
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# --- Detail Permintaan Data ---
PRODUCT_ID = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
REGION = [32.0, 44.0, 12.0, 30.0] 
START_DATE = '2007-01-01'
END_DATE = '2021-12-31'
VARIABLES = ['zos', 'thetao'] 

# --- Proses Pengunduhan ---
print(">>> Memulai proses pengunduhan data CMEMS...")

# 💡 PENYEMPURNAAN: Hapus file lama secara manual sebelum mengunduh
if os.path.exists(FULL_OUTPUT_PATH):
    print(f"    Menemukan file lama. Menghapus: {FULL_OUTPUT_PATH}")
    os.remove(FULL_OUTPUT_PATH)

print(f"    Produk    : {PRODUCT_ID}")
print(f"    Wilayah   : Lon({REGION[0]} to {REGION[1]}), Lat({REGION[2]} to {REGION[3]})")
print(f"    Periode   : {START_DATE} s.d. {END_DATE}")
print(f"    Menyimpan ke : {FULL_OUTPUT_PATH}")

try:
    cm.subset(
        username=USERNAME_CMEMS,
        password=PASSWORD_CMEMS,
        dataset_id=PRODUCT_ID,
        minimum_longitude=REGION[0],
        maximum_longitude=REGION[1],
        minimum_latitude=REGION[2],
        maximum_latitude=REGION[3],
        start_datetime=f'{START_DATE}T00:00:00',
        end_datetime=f'{END_DATE}T23:59:59',
        variables=VARIABLES,
        output_directory=OUTPUT_DIRECTORY,
        output_filename=OUTPUT_FILENAME
        # 🗑️ PARAMETER 'overwrite_output_data' DIHAPUS KARENA TIDAK DIDUKUNG
    )
    print(f"\n✅ Sukses! Data berhasil diunduh.")

except Exception as e:
    print(f"\n❌ Gagal mengunduh data. Error: {e}", file=sys.stderr)
    print("    Pastikan username, password, dan nama produk sudah benar.", file=sys.stderr)
    sys.exit(1)

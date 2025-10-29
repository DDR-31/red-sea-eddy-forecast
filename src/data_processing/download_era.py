import cdsapi
import os
import sys

# --- Pathing (Robust) ---
# Menentukan path root proyek secara dinamis
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_DIRECTORY = os.path.join(PROJECT_ROOT, 'data', 'raw')
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True) # Pastikan folder /data/raw/ ada

# --- Pengaturan Unduhan ---
start_year_total = 2007
end_year_total = 2021
step = 3 # Mengunduh dalam potongan 10 tahun

client = cdsapi.Client()

print(f"Memulai proses unduhan data ERA5 dari {start_year_total} hingga {end_year_total}...")

for start_year in range(start_year_total, end_year_total + 1, step):
    end_year = min(start_year + step - 1, end_year_total)
    years_chunk = [str(y) for y in range(start_year, end_year + 1)]
    
    # Path output yang disesuaikan dengan struktur proyek kita
    output_filename = os.path.join(
        OUTPUT_DIRECTORY, 
        f"era5_data_{start_year}-{end_year}.nc" # Disimpan di /data/raw/
    )

    if os.path.exists(output_filename):
        print(f"⚠️  File {output_filename} sudah ada, lewati...")
        continue
    
    print(f"\n---> Memproses request untuk tahun: {start_year} - {end_year}")
    print(f"     File output akan disimpan sebagai: {output_filename}")

    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": "reanalysis",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": years_chunk,
        "month": [f'{m:02d}' for m in range(1, 13)],
        "day": [f'{d:02d}' for d in range(1, 32)],
        "time": ["12:00"],
        "area": [30, 32, 12, 44], # Format: N/W/S/E
        "grid": [0.125, 0.125],
        
        # --- Parameter dikembalikan sesuai permintaan ---
        "format": "netcdf", # Menggantikan 'data_format'
        "download_format": "unarchived", # Tetap disertakan
    }

    # Untuk menghindari kebingungan, parameter 'format' adalah cara modern
    # untuk menggantikan 'data_format'. Kita akan gunakan 'format'.

    try:
        # Mengirim request ke server
        print("     Mengirim permintaan ke server CDS. Harap tunggu...")
        client.retrieve(dataset, request, output_filename)
        print(f"✅ Sukses! Request untuk {start_year}-{end_year} telah selesai.")

    except Exception as e:
        print(f"\n❌ Gagal mengunduh {start_year}-{end_year}. Error: {e}", file=sys.stderr)
        print("    Pastikan file ~/.cdsapirc sudah benar.", file=sys.stderr)
        # Hapus file parsial jika gagal agar bisa diunduh ulang
        if os.path.exists(output_filename):
            os.remove(output_filename)
        sys.exit(1) # Hentikan skrip jika ada kegagalan

print("\n=============================================")
print("Semua proses unduhan ERA5 telah selesai!")
print(f"Silakan periksa file .nc di dalam folder '{OUTPUT_DIRECTORY}'.")
print("=============================================")

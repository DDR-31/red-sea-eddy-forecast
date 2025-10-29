import xarray as xr
import os
import json
import sys

# --- Pathing ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_dataset_with_bathy.nc')
STATS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'normalization_stats.json')

print(">>> Memulai kalkulasi statistik normalisasi...")

try:
    # Buka dataset
    ds = xr.open_dataset(DATA_PATH, chunks={'time': 100})
    print(f"    Dataset dimuat dari: {DATA_PATH}")
except FileNotFoundError:
    print(f"❌ Error: File {DATA_PATH} tidak ditemukan!", file=sys.stderr)
    sys.exit(1)

stats = {}
variables_to_normalize = ['sla', 'thetao', 'u10', 'v10', 'bathy']

print("    Menghitung mean dan std... (Ini mungkin perlu beberapa saat)")

for var in variables_to_normalize:
    # .compute() akan memicu kalkulasi Dask yang efisien memori
    mean_val = ds[var].mean().compute().item()
    std_val = ds[var].std().compute().item()
    
    stats[var] = {
        'mean': mean_val,
        'std': std_val
    }
    print(f"    - {var}: mean={mean_val:.4f}, std={std_val:.4f}")

# Simpan statistik ke file JSON
with open(STATS_PATH, 'w') as f:
    json.dump(stats, f, indent=4)

print(f"\n✅ Sukses! Statistik normalisasi disimpan ke:")
print(f"   {STATS_PATH}")

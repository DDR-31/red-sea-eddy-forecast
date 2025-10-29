'''
import tensorflow as tf
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.models import Model
import os

# --- Konfigurasi Model ---
# Kita akan menggunakan 5 hari data...
N_TIME_STEPS = 5 
# ...untuk memprediksi 1 hari ke depan.

# Dimensi grid kita (dari file final_dataset.nc)
HEIGHT = 217
WIDTH = 145
# 4 channel input: sla, thetao, u10, v10
N_CHANNELS = 5

def build_convlstm_model(
    input_shape=(N_TIME_STEPS, HEIGHT, WIDTH, N_CHANNELS)
):
    """
    Membangun arsitektur model ConvLSTM.
    Input: (n_sampel, n_waktu, tinggi, lebar, n_channel)
    Output: (n_sampel, tinggi, lebar, 1) 
    """
    print(f"Membangun model dengan input shape: {input_shape}")
    
    # Definisikan input layer
    inputs = Input(shape=input_shape)

    # --- Jantung Model: Stacked ConvLSTM ---
    
    # Layer ConvLSTM pertama
    
    #'return_sequences=True' berarti layer ini akan mengeluarkan output 
    # untuk setiap timestep, untuk diteruskan ke layer berikutnya.
    x = ConvLSTM2D(
        filters=64,             # 64 filter konvolusi
        kernel_size=(3, 3),     # Ukuran filter 3x3
        padding='same',         # Pertahankan ukuran gambar
        return_sequences=True,
    )(inputs)
    x = BatchNormalization()(x) # Menstabilkan training

    # Layer ConvLSTM kedua
    
    'return_sequences=False' berarti layer ini HANYA mengeluarkan
    output dari timestep terakhir.
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False, # Hanya output di t=terakhir
    )(x)
    x = BatchNormalization()(x)

    # --- Output Layer ---
    # Kita perlu memprediksi peta 2D (tinggi, lebar) dengan 1 channel (sla).
    # Layer Conv2D 1x1 ini sempurna untuk "mengecilkan" 64 filter
    # dari layer sebelumnya menjadi 1 filter output.
    # Aktivasi 'linear' karena ini adalah masalah REGRESI (bukan klasifikasi).
    outputs = Conv2D(
        filters=1,              # 1 channel output (hanya sla)
        kernel_size=(1, 1),     # Filter 1x1
        activation='linear',    # Untuk regresi
        padding='same',
    )(x)

    # Gabungkan input dan output menjadi sebuah model
    model = Model(inputs=inputs, outputs=outputs)

    # Kompilasi model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse' # Mean Squared Error, standar untuk regresi
    )
    
    return model

if __name__ == "__main__":
    # Bagian ini hanya akan berjalan jika kamu menjalankan skrip ini secara langsung
    # Ini adalah "tes" untuk memastikan model kita berhasil dibuat
    
    print(">>> Melakukan tes pembangunan model...")
    model = build_convlstm_model()
    
    print("\n✅ Sukses! Arsitektur model berhasil dibuat.")
    print("--- Ringkasan Model ---")
    model.summary()
    
    # Simpan ringkasan ke file
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"\nRingkasan model juga disimpan di: {REPORT_DIR}/model_summary.txt")
'''

import tensorflow as tf
# 💡 Import Lambda and Reshape layers
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling3D, UpSampling3D, Concatenate, Lambda, Reshape
from tensorflow.keras.models import Model
import os

# --- Model Configuration ---
N_TIME_STEPS = 5
HEIGHT = 217
WIDTH = 145
N_CHANNELS = 5 # Now includes bathymetry

def build_unet_convlstm_model(
    input_shape=(N_TIME_STEPS, HEIGHT, WIDTH, N_CHANNELS)
):
    """
    Builds a U-Net architecture using ConvLSTM layers (v1.1 - Keras Ops Fix).
    Input: (batch, time, height, width, channels)
    Output: (batch, height, width, 1) - predicting SLA at t+1
    """
    print(f"Building U-Net ConvLSTM model with input shape: {input_shape}")

    inputs = Input(shape=input_shape)

    # --- Contracting Path (Encoder) ---
    # Level 1
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, name='conv1a')(inputs)
    conv1 = BatchNormalization(name='bn1a')(conv1)
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, name='conv1b')(conv1)
    conv1 = BatchNormalization(name='bn1b')(conv1)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2), name='pool1')(conv1) # Pool spatial dimensions

    # Level 2
    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, name='conv2a')(pool1)
    conv2 = BatchNormalization(name='bn2a')(conv2)
    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, name='conv2b')(conv2)
    conv2 = BatchNormalization(name='bn2b')(conv2)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2), name='pool2')(conv2)

    # Bottleneck (Level 3)
    conv3 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True, name='conv3a')(pool2)
    conv3 = BatchNormalization(name='bn3a')(conv3)
    # Output of bottleneck should not have time sequence
    conv3 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=False, name='conv3b')(conv3)
    conv3 = BatchNormalization(name='bn3b')(conv3)
    # Output shape: (batch, height/4, width/4, 128)

    # --- Expanding Path (Decoder) ---

    # Level 2 Upsample
    # 💡 FIX: Use Reshape and UpSampling3D correctly
    # Add a time dimension of 1 for UpSampling3D
    up2_input = Reshape((1, conv3.shape[1], conv3.shape[2], conv3.shape[3]), name='reshape_up2_in')(conv3)
    up2 = UpSampling3D(size=(1, 2, 2), name='upsample2')(up2_input)
    # Remove the time dimension
    up2 = Reshape((up2.shape[2], up2.shape[3], up2.shape[4]), name='reshape_up2_out')(up2)
    # Output shape: (batch, height/2, width/2, 128)

    # 💡 FIX: Use Lambda layer for skip connection slicing
    skip2 = Lambda(lambda x: x[:, -1, :, :, :], name='skip2_slice')(conv2) # Get last time step
    # Output shape: (batch, height/2, width/2, 64)

    concat2 = Concatenate(name='concat2')([up2, skip2])
    # Output shape: (batch, height/2, width/2, 128 + 64)

    dconv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='dconv2a')(concat2)
    dconv2 = BatchNormalization(name='dbn2a')(dconv2)
    dconv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='dconv2b')(dconv2)
    dconv2 = BatchNormalization(name='dbn2b')(dconv2)
    # Output shape: (batch, height/2, width/2, 64)


    # Level 1 Upsample
    # 💡 FIX: Use Reshape and UpSampling3D correctly
    up1_input = Reshape((1, dconv2.shape[1], dconv2.shape[2], dconv2.shape[3]), name='reshape_up1_in')(dconv2)
    up1 = UpSampling3D(size=(1, 2, 2), name='upsample1')(up1_input)
    up1 = Reshape((up1.shape[2], up1.shape[3], up1.shape[4]), name='reshape_up1_out')(up1)
    # Output shape up1: (batch, 216, 144, 64)

    # skip1 tetap sama
    skip1 = Lambda(lambda x: x[:, -1, :, :, :], name='skip1_slice')(conv1) # Get last time step
    # Output shape skip1: (batch, 217, 145, 32)

    # 💡 FIX: Crop skip1 agar ukurannya sama dengan up1
    # Cropping2D((top_crop, bottom_crop), (left_crop, right_crop))
    # Kita perlu buang 1 dari bawah (217->216) dan 1 dari kanan (145->144)
    skip1_cropped = tf.keras.layers.Cropping2D(cropping=((0, 1), (0, 1)), name='crop_skip1')(skip1)
    # Output shape skip1_cropped: (batch, 216, 144, 32)

    # 💡 FIX: Concatenate up1 dengan skip1_cropped
    concat1 = Concatenate(name='concat1')([up1, skip1_cropped])
    # Output shape concat1: (batch, 216, 144, 64 + 32)

    # dconv1 tetap sama
    dconv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', name='dconv1a')(concat1)
    dconv1 = BatchNormalization(name='dbn1a')(dconv1)
    dconv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', name='dconv1b')(dconv1)
    dconv1 = BatchNormalization(name='dbn1b')(dconv1)
    # Output shape dconv1: (batch, 216, 144, 32)

    # --- Output Layer ---
    # 💡 FIX: Output layer sekarang menerima input 216x144
    # Karena padding='same', outputnya juga akan 216x144
    outputs_precrop = Conv2D(
        filters=1, kernel_size=(1, 1), activation='linear', padding='same', name='output_conv'
    )(dconv1)
    # Output shape: (batch, 216, 144, 1)

    # 💡 FIX: Kita perlu padding agar output kembali ke 217x145
    # Gunakan ZeroPadding2D untuk menambahkan 1 baris nol di bawah dan 1 kolom nol di kanan
    outputs = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='output_padding')(outputs_precrop)
    # Output shape final: (batch, 217, 145, 1)


    # Build and compile (tetap sama)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )

    return model

# Keep the if __name__ == "__main__": block the same
if __name__ == "__main__":
    print(">>> Performing model build test...")
    model = build_unet_convlstm_model()
    print("\n✅ Success! U-Net ConvLSTM model built.")
    print("--- Model Summary ---")
    model.summary()

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, 'unet_model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"\nModel summary also saved to: {REPORT_DIR}/unet_model_summary.txt")

# Deteksi-Anomali-IoT-Lightgbm

## Langkah-langkah Penggunaan

### 1. Menyiapkan environment

1. **Buat virtual environment**:
    ```bash
    python -m venv .venv
    ```

2. **Aktifkan virtual environment**:
    - **Windows**:
        ```bash
        .venv\Scripts\activate
        ```
    - **Mac/Linux**:
        ```bash
        source .venv/bin/activate
        ```

3. **Instal semua dependensi**:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Melatih Model

Jika ingin melatih ulang model, gunakan script `model_training.py` atau `model_trainingv2.py` yang ada di folder `scripts/`.

1. **Menjalankan script pelatihan**:
    ```bash
    python scripts/model_training.py
    ```

### 3. Menjalankan Server

Jalankan server menggunakan script `server.py`.

1. **Menjalankan server**:
    ```bash
    python scripts/server.py
    ```

### 4. Menggunakan Klien untuk Prediksi

Setelah server berjalan, gunakan `client.py` untuk mengirim data dan mendapatkan prediksi.

1. **Jalankan klien**:
    ```bash
    python scripts/client.py
    ```

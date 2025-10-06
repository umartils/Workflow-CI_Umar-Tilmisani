#!/bin/bash

echo "🚀 Menjalankan CI Training MLflow Project"

# Pastikan error menghentikan eksekusi
set -e

# 1️⃣ Buat / perbarui environment conda
echo "[1/3] Mengatur environment conda..."
if conda env list | grep -q "mlflow-env"; then
    echo "Environment 'mlflow-env' sudah ada, memperbarui..."
    conda env update -f conda.yaml --prune
else
    echo "Membuat environment baru 'mlflow-env'..."
    conda env create -f conda.yaml
fi

# 2️⃣ Aktifkan environment
echo "[2/3] Mengaktifkan environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlflow-env

# 3️⃣ Jalankan MLflow project
echo "[3/3] Menjalankan MLflow training..."
mlflow run ./MLProject -P n_estimators=100 -P max_depth=20

echo "✅ Training selesai dengan sukses!"

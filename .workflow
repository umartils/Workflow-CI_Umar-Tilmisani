#!/bin/bash

echo "Menjalankan CI Training MLflow Project"

# Hentikan eksekusi jika ada error
set -e

echo "Mengatur environment conda..."
if conda env list | grep -q "mlflow-env"; then
    echo "Environment 'mlflow-env' sudah ada, memperbarui..."
    conda env update -f MLProject/conda.yaml --prune
else
    echo "Membuat environment baru 'mlflow-env'..."
    conda env create -f MLProject/conda.yaml
fi

echo "Mengaktifkan environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlflow-env

echo "Menjalankan MLflow training..."
cd MLProject
mlflow run . -P n_estimators=100 -P max_depth=20

echo "========================================"
echo "Training selesai dengan sukses!"
echo "========================================"

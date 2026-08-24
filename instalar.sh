#!/bin/bash
# instalar.sh - Motor de deteccion propio (birdnet-lsd), instalacion
# aislada: venv de Python dedicado + modelo v10 (LSDTector-BirdNET-retrain-bsas)
# + sesgo regional (LSDTector-BirdNET-custom-v1), descargados via
# raw.githubusercontent.com, mismo patron que actualizar_modelo.sh de
# LSD-Tector2.0. No requiere git ni credenciales en el dispositivo.
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
MODELO_DIR="$SCRIPT_DIR/modelo"

echo "==> Paso 1/3: entorno virtual de Python ($VENV_DIR)"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$SCRIPT_DIR/requirements.txt"

echo "==> Paso 2/3: modelo v10 (LSDTector-BirdNET-retrain-bsas)"
mkdir -p "$MODELO_DIR"
REPO_MODELO="LSDArroyoGold/LSDTector-BirdNET-retrain-bsas"
RAW_MODELO="https://raw.githubusercontent.com/$REPO_MODELO/main"
curl -sf -o "$MODELO_DIR/LSDTector_Classifier_v2.tflite" "$RAW_MODELO/modelo/LSDTector_Classifier_v2.tflite"
curl -sf -o "$MODELO_DIR/LSDTector_Classifier_v2_Labels.txt" "$RAW_MODELO/modelo/LSDTector_Classifier_v2_Labels.txt"
# v10_podado.json ya viene commiteado en este mismo repo (modelo/v10_podado.json,
# documenta que neuronas quedan suprimidas) -- no se pisa, es propio de este repo.

echo "==> Paso 3/3: sesgo regional (LSDTector-BirdNET-custom-v1)"
REPO_BIAS="LSDArroyoGold/LSDTector-BirdNET-custom-v1"
RAW_BIAS="https://raw.githubusercontent.com/$REPO_BIAS/main"
curl -sf -o "$MODELO_DIR/stock_regional_meta.json" "$RAW_BIAS/model/stock_regional_meta.json"

echo ""
echo "==> Instalacion completa. Verificar con:"
echo "    source $VENV_DIR/bin/activate"
echo "    python3 $SCRIPT_DIR/pruebas/verificar_instalacion.py"

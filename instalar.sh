#!/bin/bash
# instalar.sh - Motor de deteccion propio (birdnet-lsd), instalacion
# aislada: venv de Python dedicado. Modelo cambiado el 29/08/2026 --
# BirdNET reentrenado (tflite, descargado de otro repo via
# raw.githubusercontent.com) retirado por completo, reemplazado por
# TectorNet: Perch 2.0 ONNX + BirdSet EfficientNetB1 ONNX. BirdSet ya
# viene commiteado en este mismo repo (modelo/birdset_efficientnetb1.onnx,
# no tiene host publico propio todavia); Perch2 se descarga solo, la
# primera vez que corre motor.py (cache local de huggingface_hub, no
# hace falta bajarlo aca).
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "==> Paso 1/2: entorno virtual de Python ($VENV_DIR)"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$SCRIPT_DIR/requirements.txt"

echo "==> Paso 2/2: verificando que el clasificador carga y clasifica (descarga Perch2 de HuggingFace la primera vez)"
python3 "$SCRIPT_DIR/scripts/verificar_clasificador.py"

echo ""
echo "==> Instalacion completa. Para correr el motor en vivo (necesita arecord/ALSA):"
echo "    source $VENV_DIR/bin/activate"
echo "    python3 $SCRIPT_DIR/scripts/motor.py"

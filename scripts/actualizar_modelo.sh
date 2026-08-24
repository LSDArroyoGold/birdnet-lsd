#!/bin/bash
#
# actualizar_modelo.sh - Mantiene al dia el modelo v10
# (LSDTector-BirdNET-retrain-bsas) y el sesgo regional
# (LSDTector-BirdNET-custom-v1) que usa birdnet-lsd, sin tocar el resto
# del repo. Mismo patron que actualizar_repo.sh/actualizar_modelo.sh de
# LSD-Tector1.1/2.0: compara el SHA del ultimo commit via la API de
# GitHub, y si cambio, descarga via raw.githubusercontent.com. No
# requiere git ni credenciales en el dispositivo, solo HTTPS de salida.
#
# instalar.sh descarga el modelo y el sesgo UNA vez, al instalar -- este
# script es lo que los mantiene al dia despues, sin reinstalar nada a
# mano. Reinicia birdnet-lsd.service solo si de verdad hubo un cambio.

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
BASE_PATH="$(dirname "$SCRIPT_DIR")"
MODELO_DIR="$BASE_PATH/modelo"

REPO_MODELO="LSDArroyoGold/LSDTector-BirdNET-retrain-bsas"
RAW_MODELO="https://raw.githubusercontent.com/$REPO_MODELO/main"
API_MODELO="https://api.github.com/repos/$REPO_MODELO/commits/main"
MARCA_MODELO="$BASE_PATH/.ultima_actualizacion_modelo"

REPO_BIAS="LSDArroyoGold/LSDTector-BirdNET-custom-v1"
RAW_BIAS="https://raw.githubusercontent.com/$REPO_BIAS/main"
API_BIAS="https://api.github.com/repos/$REPO_BIAS/commits/main"
MARCA_BIAS="$BASE_PATH/.ultima_actualizacion_bias"

HUBO_CAMBIO=0

# --- Modelo v10 ---
SHA_ACTUAL_MODELO=$(curl -s "$API_MODELO" | python3 -c "import sys,json; print(json.load(sys.stdin)['sha'])" 2>/dev/null)
ULTIMO_SHA_MODELO=$(cat "$MARCA_MODELO" 2>/dev/null)

if [ -n "$SHA_ACTUAL_MODELO" ] && [ "$SHA_ACTUAL_MODELO" != "$ULTIMO_SHA_MODELO" ]; then
	TMP="$BASE_PATH/.actualizar_modelo_tmp"
	rm -rf "$TMP"
	mkdir -p "$TMP"
	if curl -sf -o "$TMP/modelo.tflite" "$RAW_MODELO/modelo/LSDTector_Classifier_v2.tflite" \
	   && curl -sf -o "$TMP/modelo_labels.txt" "$RAW_MODELO/modelo/LSDTector_Classifier_v2_Labels.txt"; then
		# Mismo filesystem que MODELO_DIR: el mv es un rename atomico, no
		# deja al modelo a medio escribir si algo se corta en el medio.
		mv "$TMP/modelo.tflite" "$MODELO_DIR/LSDTector_Classifier_v2.tflite"
		mv "$TMP/modelo_labels.txt" "$MODELO_DIR/LSDTector_Classifier_v2_Labels.txt"
		echo "$SHA_ACTUAL_MODELO" > "$MARCA_MODELO"
		HUBO_CAMBIO=1
		echo "Modelo v10 actualizado ($SHA_ACTUAL_MODELO)"
	else
		echo "Fallo la descarga del modelo v10, aborto sin tocar el actual" >&2
	fi
	rm -rf "$TMP"
fi

# --- Sesgo regional ---
SHA_ACTUAL_BIAS=$(curl -s "$API_BIAS" | python3 -c "import sys,json; print(json.load(sys.stdin)['sha'])" 2>/dev/null)
ULTIMO_SHA_BIAS=$(cat "$MARCA_BIAS" 2>/dev/null)

if [ -n "$SHA_ACTUAL_BIAS" ] && [ "$SHA_ACTUAL_BIAS" != "$ULTIMO_SHA_BIAS" ]; then
	TMP_BIAS="$BASE_PATH/.actualizar_bias_tmp.json"
	if curl -sf -o "$TMP_BIAS" "$RAW_BIAS/model/stock_regional_meta.json"; then
		mv "$TMP_BIAS" "$MODELO_DIR/stock_regional_meta.json"
		echo "$SHA_ACTUAL_BIAS" > "$MARCA_BIAS"
		HUBO_CAMBIO=1
		echo "Sesgo regional actualizado ($SHA_ACTUAL_BIAS)"
	else
		echo "Fallo la descarga del sesgo regional, aborto sin tocar el actual" >&2
		rm -f "$TMP_BIAS"
	fi
fi

if [ "$HUBO_CAMBIO" = "1" ] && systemctl list-unit-files birdnet-lsd.service &>/dev/null; then
	sudo systemctl restart birdnet-lsd.service
fi

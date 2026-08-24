#!/bin/bash
#
# migrar_a_birdnet_lsd.sh - Instala y activa birdnet-lsd en un dispositivo
# LSD-Tector que todavia corre BirdNET-Pi stock, disparado por el ciclo
# de auto-actualizacion normal del dispositivo (inicio_amanecer.sh /
# inicio_atardecer.sh) -- pensado para correr desatendido, en un
# dispositivo en el campo sin acceso SSH mientras corre.
#
# Uso: migrar_a_birdnet_lsd.sh <DRIVE_PATH> <DRIVE_SUBCARPETA>
#   DRIVE_PATH: carpeta de Drive del dispositivo (ej. "Laboratorio 6" o
#     "LSD-Tector"), la misma que ya usan sus scripts de sincronizacion.
#   DRIVE_SUBCARPETA: subcarpeta de detecciones dentro de DRIVE_PATH que
#     el dispositivo ya viene usando (ej. "BirdNET_Detecciones" o
#     "Detecciones") -- para no partir su historial en dos lugares.
#
# Principio rector: NUNCA dejar al dispositivo sin motor de deteccion.
# BirdNET-Pi stock sigue activo en todo momento hasta que birdnet-lsd.
# service se confirma sano (activo Y con arecord realmente capturando
# audio) -- recien ahi se hace el swap. Si cualquier paso falla, se
# aborta sin tocar nada mas: BirdNET-Pi sigue andando como si esto nunca
# hubiera corrido, y el proximo ciclo (siguiente ventana) reintenta desde
# cero (idempotente: git pull en vez de clone si ya existe, instalar.sh
# se saltea si el venv ya esta armado).
#
# Una vez que la migracion se completa OK, no se vuelve a intentar (marca
# con /home/lsd/.birdnet_lsd_migrado) -- correrlo de nuevo despues de eso
# no hace nada.

set -uo pipefail  # sin -e a proposito: cada paso se chequea a mano para poder abortar limpio

DRIVE_PATH="${1:?Uso: migrar_a_birdnet_lsd.sh <DRIVE_PATH> <DRIVE_SUBCARPETA>}"
DRIVE_SUBCARPETA="${2:?Uso: migrar_a_birdnet_lsd.sh <DRIVE_PATH> <DRIVE_SUBCARPETA>}"

MARCA_MIGRADO="/home/lsd/.birdnet_lsd_migrado"
BIRDNET_CONF="/home/lsd/BirdNET-Pi/birdnet.conf"
BIRDNET_LSD_DIR="/home/lsd/birdnet-lsd"

log() {
	python3 /home/lsd/log_sistema.py MSG "birdnet-lsd: $1" 2>/dev/null \
		|| python3 /home/lsd/python/log_sistema.py MSG "birdnet-lsd: $1" 2>/dev/null \
		|| echo "birdnet-lsd: $1"
}

abortar() {
	log "migracion abortada -- $1. BirdNET-Pi stock sigue activo, sin cambios."
	exit 1
}

if [ -f "$MARCA_MIGRADO" ]; then
	exit 0
fi

[ -f "$BIRDNET_CONF" ] || abortar "no se encontro $BIRDNET_CONF (BirdNET-Pi no esta instalado)"

BIRDWEATHER_ID=$(awk -F= '/^BIRDWEATHER_ID=/{print $2}' "$BIRDNET_CONF" | tr -d ' \r')
LATITUDE=$(awk -F= '/^LATITUDE=/{print $2}' "$BIRDNET_CONF" | tr -d ' \r')
LONGITUDE=$(awk -F= '/^LONGITUDE=/{print $2}' "$BIRDNET_CONF" | tr -d ' \r')
REC_CARD=$(awk -F= '/^REC_CARD=/{print $2}' "$BIRDNET_CONF" | tr -d ' \r')
CHANNELS=$(awk -F= '/^CHANNELS=/{print $2}' "$BIRDNET_CONF" | tr -d ' \r')

[ -n "$LATITUDE" ] && [ -n "$LONGITUDE" ] || abortar "no se pudieron leer LATITUDE/LONGITUDE de $BIRDNET_CONF"
[ -n "$REC_CARD" ] || REC_CARD="default"
[ -n "$CHANNELS" ] || CHANNELS="2"

# --- clonar/actualizar birdnet-lsd ---
if [ ! -d "$BIRDNET_LSD_DIR" ]; then
	git clone https://github.com/LSDArroyoGold/birdnet-lsd.git "$BIRDNET_LSD_DIR" \
		|| abortar "fallo el clone de birdnet-lsd"
else
	git -C "$BIRDNET_LSD_DIR" pull origin main || abortar "fallo git pull de birdnet-lsd"
fi

cd "$BIRDNET_LSD_DIR" || abortar "no se pudo entrar a $BIRDNET_LSD_DIR"

# --- instalar (venv + modelo + sesgo), solo si todavia no esta ---
# Chequea el ultimo archivo que instalar.sh deja (no solo la carpeta
# venv/): si una corrida anterior fallo a mitad de camino (ej. se corto
# la red bajando el modelo), venv/ ya existiria pero incompleto, y
# saltear instalar.sh de nuevo lo dejaria asi para siempre.
if [ ! -f modelo/stock_regional_meta.json ]; then
	bash instalar.sh || abortar "fallo instalar.sh"
fi

# --- configs especificas de este dispositivo, derivadas de birdnet.conf ---
cat > config/config_birdweather.txt <<EOF
BIRDWEATHER_ID = $BIRDWEATHER_ID
LATITUDE = $LATITUDE
LONGITUDE = $LONGITUDE
EOF
chmod 600 config/config_birdweather.txt

cat > config/config_sincronizacion.txt <<EOF
RCLONE_CONFIG = /home/lsd/.config/rclone/rclone.conf
DRIVE_REMOTE = gdrive
DRIVE_PATH = $DRIVE_PATH
DRIVE_SUBCARPETA = $DRIVE_SUBCARPETA
AUDIO_ROOT = /home/lsd/BirdSongs/Extracted
REC_CARD = $REC_CARD
CHANNELS = $CHANNELS
EOF
chmod 600 config/config_sincronizacion.txt

# --- servicio systemd (y logrotate, linger) ---
bash instalar_servicio.sh || abortar "fallo instalar_servicio.sh"

sudo systemctl restart birdnet-lsd.service || abortar "no arranco birdnet-lsd.service"

# --- verificacion de salud ANTES de tocar BirdNET-Pi ---
# Tiempo generoso: carga de TensorFlow + modelo puede tardar en hardware
# mas viejo/lento que el que se uso para desarrollar esto.
sleep 40

if ! systemctl is-active --quiet birdnet-lsd.service; then
	abortar "birdnet-lsd.service no quedo activo despues de arrancar"
fi

# No alcanza con que el proceso este activo -- confirmar que arecord
# realmente esta capturando audio (el bug de XDG_RUNTIME_DIR/PulseAudio
# encontrado el 23/08 en tector2 dejaba el proceso "activo" pero
# reintentando arecord en loop, sin audio real).
if ! pgrep -f "arecord -f S16_LE" > /dev/null; then
	sudo systemctl stop birdnet-lsd.service
	abortar "birdnet-lsd.service esta activo pero arecord no esta corriendo (problema de audio/microfono)"
fi

# --- swap: recien ahora se apaga BirdNET-Pi stock ---
sudo systemctl stop birdnet_recording.service birdnet_analysis.service 2>/dev/null
sudo systemctl disable birdnet_recording.service birdnet_analysis.service 2>/dev/null

touch "$MARCA_MIGRADO"
log "migracion a birdnet-lsd completada OK (BirdWeather=$BIRDWEATHER_ID, card=$REC_CARD, canales=$CHANNELS)"

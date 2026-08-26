#!/bin/bash
#
# actualizar_birdnet_lsd.sh - Actualiza el checkout de birdnet-lsd ya
# migrado (git pull) y reinicia el servicio si hubo cambios, pensado
# para correr en cada ventana (amanecer/atardecer) de un dispositivo en
# el campo sin acceso SSH mientras corre.
#
# Por que existe: migrar_a_birdnet_lsd.sh (el clone + instalacion inicial)
# solo corre mientras no exista /home/lsd/.birdnet_lsd_migrado -- una vez
# migrado, ese bloque entero se salta para siempre en inicio_amanecer.sh/
# inicio_atardecer.sh, y con el se salteaba tambien cualquier chance de
# bajar cambios nuevos. Encontrado el 26/08: dos fixes reales (CONFIDENCE
# 0.6->0.5, formato de hora en el nombre de archivo) quedaron pusheados
# en GitHub sin ninguna forma de llegar a tector1.
#
# Mismo principio rector que migrar_a_birdnet_lsd.sh: NUNCA dejar al
# dispositivo sin motor de deteccion sano. Si el pull trae un cambio que
# rompe el servicio (no queda activo, o arecord no arranca), se revierte
# solo al commit anterior (que ya se sabia sano) y se reintenta -- nunca
# se deja el pull nuevo aplicado si no paso el chequeo de salud.

set -uo pipefail

BIRDNET_LSD_DIR="/home/lsd/birdnet-lsd"
MARCA_MIGRADO="/home/lsd/.birdnet_lsd_migrado"

log() {
	python3 /home/lsd/log_sistema.py MSG "birdnet-lsd: $1" 2>/dev/null \
		|| python3 /home/lsd/python/log_sistema.py MSG "birdnet-lsd: $1" 2>/dev/null \
		|| echo "birdnet-lsd: $1"
}

# Solo tiene sentido correr esto despues de una migracion ya completa --
# antes de eso, migrar_a_birdnet_lsd.sh se encarga del clone inicial.
[ -f "$MARCA_MIGRADO" ] || exit 0
[ -d "$BIRDNET_LSD_DIR" ] || exit 0

cd "$BIRDNET_LSD_DIR" || exit 0

SHA_ANTERIOR=$(git rev-parse HEAD 2>/dev/null) || exit 0

git fetch --quiet origin main || { log "ALERTA: git fetch fallo, sigue con la version actual"; exit 0; }

SHA_REMOTO=$(git rev-parse origin/main 2>/dev/null)
if [ -z "$SHA_REMOTO" ] || [ "$SHA_REMOTO" = "$SHA_ANTERIOR" ]; then
	exit 0  # sin cambios, nada que hacer
fi

# Si requirements.txt cambio entre ambos commits, reinstalar dependencias
# antes de reiniciar el servicio (si no cambio, no hace falta tocar el venv).
if ! git diff --quiet "$SHA_ANTERIOR" "$SHA_REMOTO" -- requirements.txt 2>/dev/null; then
	"$BIRDNET_LSD_DIR/venv/bin/pip" install -q -r "$BIRDNET_LSD_DIR/requirements.txt" 2>/dev/null
fi

git reset --hard "$SHA_REMOTO" --quiet

chequear_salud() {
	sudo systemctl restart birdnet-lsd.service
	sleep 5
	systemctl is-active --quiet birdnet-lsd.service || return 1
	pgrep -f "arecord -f S16_LE" > /dev/null || return 1
	return 0
}

if chequear_salud; then
	log "actualizado de ${SHA_ANTERIOR:0:7} a ${SHA_REMOTO:0:7}, servicio sano"
else
	log "ALERTA: ${SHA_REMOTO:0:7} rompio el servicio (revirtiendo a ${SHA_ANTERIOR:0:7})"
	git reset --hard "$SHA_ANTERIOR" --quiet
	if chequear_salud; then
		log "revertido a ${SHA_ANTERIOR:0:7} OK, servicio sano de nuevo"
	else
		log "ALERTA CRITICA: servicio no sano ni en ${SHA_REMOTO:0:7} ni revertido a ${SHA_ANTERIOR:0:7} -- revisar en persona"
	fi
fi

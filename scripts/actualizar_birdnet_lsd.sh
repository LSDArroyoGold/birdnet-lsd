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
# rompe el servicio (no queda activo, o arecord no arranca, o el
# clasificador no clasifica), se revierte solo al commit anterior (que ya
# se sabia sano) y se reintenta -- nunca se deja el pull nuevo aplicado
# si no paso el chequeo de salud.
#
# Verificado en campo en tector1 el 29/08/2026: el mecanismo de re-exec
# (ver "Fase 1"/"Fase 2" mas abajo) no protege la transicion desde una
# version del script ANTERIOR a que este mismo mecanismo existiera (esa
# version vieja no sabe re-ejecutarse, es un limite de arranque inherente
# a cualquier fix que un script se aplique a si mismo) -- pero a partir de
# esta version en adelante, cada actualizacion futura si pasa por el
# re-exec correctamente. Este comentario es, de hecho, el primer cambio
# real usado para confirmarlo end-to-end en tector1.

set -uo pipefail

BIRDNET_LSD_DIR="/home/lsd/birdnet-lsd"
MARCA_MIGRADO="/home/lsd/.birdnet_lsd_migrado"

log() {
	# Arreglado el 29/08/2026: log_sistema.py real (/home/lsd/log_sistema.py)
	# es especifico de eventos de ventana (INICIO/FIN/SIN_CONEXION con datos
	# de bateria PiJuice, ademas de importar y hablarle al hardware de
	# PiJuice por I2C en cada llamada) -- nunca tuvo soporte real para un
	# evento "MSG" generico, asi que la llamada anterior siempre caia en la
	# rama 'else' (fin_esperado = sys.argv[3]) y explotaba con IndexError al
	# recibir solo 2 argumentos. En la practica, las alertas de este script
	# nunca habian llegado al log real ni a Drive, solo al stdout de la
	# corrida (visible si alguien la mira en vivo, invisible en campo).
	#
	# Fix: escribir directo a log_sistema.txt (el archivo real, sin pasar
	# por ese script ni tocar hardware de PiJuice) -- mismo archivo que
	# inicio_amanecer.sh/inicio_atardecer.sh ya suben a Drive con rclone en
	# cada ventana, asi que las alertas quedan visibles sin agregar ningun
	# mecanismo de sincronizacion nuevo.
	#
	# BUG REAL encontrado el 4/9/2026 en tector2: el comentario de arriba
	# decia la verdad a MEDIAS -- "log_sistema.txt" existe en DOS lugares
	# distintos segun el dispositivo, porque este script es de un repo
	# hermano (birdnet-lsd) compartido tal cual entre los dos:
	#   - tector2 (LSD-Tector2.0): anidado, /home/lsd/LSD-Tector2.0/log_sistema.txt
	#   - tector1 (LSD-Tector1.1): plano, /home/lsd/log_sistema.txt directo
	# El fix del 29/08 escribia a la ruta plana -- resulta que esa SI es
	# la correcta para tector1 (su convencion real), pero NO para tector2
	# (donde nadie sube ese archivo suelto a ningun lado). Un fix
	# hardcodeado a la ruta anidada de tector2 arreglaria uno y romperia
	# el otro en silencio -- exactamente el mismo problema de nuevo. Se
	# detecta por la presencia de la carpeta LSD-Tector2.0 en vez de
	# asumir una sola convencion (mismo criterio que _LOG_SISTEMA_REAL en
	# motor.py, mismo bug, encontrados el mismo dia).
	local ruta_log
	if [ -d /home/lsd/LSD-Tector2.0 ]; then
		ruta_log=/home/lsd/LSD-Tector2.0/log_sistema.txt
	else
		ruta_log=/home/lsd/log_sistema.txt
	fi
	local timestamp
	timestamp=$(date '+%Y-%m-%d %H:%M')
	echo "[$timestamp] birdnet-lsd: $1" >> "$ruta_log"
	echo "birdnet-lsd: $1"
}

# Solo tiene sentido correr esto despues de una migracion ya completa --
# antes de eso, migrar_a_birdnet_lsd.sh se encarga del clone inicial.
[ -f "$MARCA_MIGRADO" ] || exit 0
[ -d "$BIRDNET_LSD_DIR" ] || exit 0

cd "$BIRDNET_LSD_DIR" || exit 0

if [ -z "${_REEXEC:-}" ]; then
	# --- Fase 1: detectar cambios, traerlos, re-ejecutar desde cero ---
	#
	# Bug real encontrado en tector1 el 29/08/2026: este mismo script vive
	# DENTRO del repo que "git reset --hard" reescribe -- si esa linea
	# corre y despues el script sigue leyendo mas lineas de SI MISMO desde
	# el mismo archivo (la definicion de chequear_salud(), mas abajo), bash
	# puede terminar leyendo una mezcla de la version vieja y la nueva del
	# archivo (confirmado empiricamente: cambios reales a chequear_salud()
	# no se aplicaban de forma consistente en la misma corrida que los
	# traia). Mismo patron ya resuelto para este problema en
	# actualizar_repo.sh de LSD-Tector2.0: en vez de seguir leyendo el
	# archivo que se acaba de reescribir, se re-ejecuta desde cero
	# (`exec`) una vez que el archivo en disco ya es estable y completo.
	SHA_ANTERIOR=$(git rev-parse HEAD 2>/dev/null) || exit 0

	git fetch --quiet origin main || { log "ALERTA: git fetch fallo, sigue con la version actual"; exit 0; }

	SHA_REMOTO=$(git rev-parse origin/main 2>/dev/null)
	if [ -z "$SHA_REMOTO" ] || [ "$SHA_REMOTO" = "$SHA_ANTERIOR" ]; then
		exit 0  # sin cambios, nada que hacer
	fi

	NECESITA_REINSTALAR=0
	if ! git diff --quiet "$SHA_ANTERIOR" "$SHA_REMOTO" -- requirements.txt 2>/dev/null; then
		NECESITA_REINSTALAR=1
	fi

	git reset --hard "$SHA_REMOTO" --quiet

	if [ "$NECESITA_REINSTALAR" = "1" ]; then
		"$BIRDNET_LSD_DIR/venv/bin/pip" install -q -r "$BIRDNET_LSD_DIR/requirements.txt" 2>/dev/null
	fi

	# _REEXEC evita un bucle si por lo que sea el archivo siguiera
	# "cambiando" -- de aca en mas, TODO lo que sigue se lee de una copia
	# fresca y completa del archivo ya actualizado.
	SHA_ANTERIOR="$SHA_ANTERIOR" _REEXEC=1 exec bash "$BIRDNET_LSD_DIR/scripts/actualizar_birdnet_lsd.sh"
fi

# --- Fase 2: chequeo de salud + revert si hace falta (siempre corre
# desde una lectura fresca del archivo, ver fase 1 arriba) ---
SHA_REMOTO=$(git rev-parse HEAD 2>/dev/null)

chequear_salud() {
	sudo systemctl restart birdnet-lsd.service

	# Espera con reintentos, no un "sleep 5" fijo -- encontrado en tector1
	# el 29/08: con el clasificador viejo (BirdNET tflite, carga casi
	# instantanea) 5s alcanzaban de sobra, pero TectorNet (Perch2+BirdSet
	# ONNX) tarda ~4-5s solo en cargar los modelos (medido esa mañana)
	# ANTES de que motor.py llegue a lanzar arecord -- confirmado que un
	# "sleep 5" fijo hacia fallar el chequeo (pgrep no encontraba arecord
	# todavia) tanto probando el commit nuevo como, con menos margen aun,
	# revirtiendo al viejo justo despues (dos restarts seguidos, con la
	# maquina bajo mas carga de la usual por el reset/pip recien hecho).
	# Sondear cada 1s es robusto a esta variacion sin depender de acertar
	# un numero fijo de antemano -- pero el numero de intentos si importa:
	# 15s (el valor original del 29/08) resulto ser insuficiente en la
	# practica. Confirmado en tector2 el 4/9/2026: bajo carga real de CPU
	# (Pi caliente por uso sostenido, no por ningun bug de este commit en
	# particular), el mismo restart que normalmente tarda 1s en levantar
	# arecord tardo mas de 15s DOS VECES SEGUIDAS -- el chequeo dio por
	# roto un commit que en realidad funcionaba bien (confirmado
	# reproduciendolo a mano: el servicio SI llegaba a levantar arecord y
	# clasificar correctamente, solo que mas tarde de lo que este poll
	# esperaba), disparando un rollback innecesario. Subido a 45s con
	# margen real para ese escenario.
	ok=0
	for _ in $(seq 1 45); do
		if systemctl is-active --quiet birdnet-lsd.service && pgrep -f "arecord -f S16_LE" > /dev/null; then
			ok=1
			break
		fi
		sleep 1
	done
	if [ "$ok" != "1" ]; then
		log "ALERTA: chequear_salud -- arecord no aparecio en 45s tras el restart (servicio activo=$(systemctl is-active birdnet-lsd.service 2>&1))"
		return 1
	fi

	# No alcanza con "el proceso esta vivo" -- un clasificador colgado o
	# demasiado lento (ver motor.py, hilo_clasificador) pasaria este
	# chequeo igual, porque la captura de audio esta desacoplada de la
	# clasificacion a proposito. Se corre una clasificacion real sobre un
	# audio de prueba fijo, con un timeout externo (no un try/except --
	# eso no protege contra un cuelgue real de onnxruntime, solo contra
	# una excepcion) para agarrar tanto crashes como cuelgues.
	#
	# HF_HOME explicito: este script corre como root (via sudo), que
	# tiene su propio $HOME (/root) sin el cache de Perch2 -- sin esto,
	# CADA corrida de este chequeo descarga Perch2 de nuevo bajo
	# /root/.cache/huggingface, que en la practica tarda mas de 60s
	# (confirmado en tector1 el 29/08: timeout real, exit code 124, sin
	# ningun problema real de codigo de por medio). El servicio real
	# (birdnet-lsd.service) corre como "lsd" (ver systemd/birdnet-lsd.service),
	# asi que apuntamos al cache de ESE usuario para que este chequeo
	# use el mismo cache ya poblado, sin importar que usuario lo invoque.
	export HF_HOME=/home/lsd/.cache/huggingface
	if ! timeout 90 "$BIRDNET_LSD_DIR/venv/bin/python3" "$BIRDNET_LSD_DIR/scripts/verificar_clasificador.py"; then
		log "ALERTA: chequear_salud -- verificar_clasificador.py fallo o colgo (90s)"
		return 1
	fi
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

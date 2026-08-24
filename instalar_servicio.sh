#!/bin/bash
# instalar_servicio.sh - registra birdnet-lsd como servicio systemd, para
# que corra en segundo plano y se reinicie solo (Restart=always) si el
# proceso muere o el dispositivo reinicia. Paso separado de instalar.sh
# a proposito: instalar.sh se puede correr en cualquier maquina (para
# desarrollo/pruebas), esto es especifico de un dispositivo en produccion.
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
UNIDAD="birdnet-lsd.service"

if [ ! -d "$SCRIPT_DIR/venv" ]; then
	echo "No existe $SCRIPT_DIR/venv -- correr primero bash instalar.sh" >&2
	exit 1
fi

UID_LSD="$(id -u lsd)"

echo "==> Habilitando linger para lsd (para que su sesion de PulseAudio -- de la"
echo "    que depende arecord via el dispositivo ALSA 'default' -- sobreviva sin"
echo "    una sesion de login activa, incluida despues de un reinicio)"
sudo loginctl enable-linger lsd

echo "==> Registrando $UNIDAD (BASE_PATH=$SCRIPT_DIR, UID_LSD=$UID_LSD)"
sed -e "s|__BASE_PATH__|$SCRIPT_DIR|g" -e "s|__UID_LSD__|$UID_LSD|g" \
	"$SCRIPT_DIR/systemd/birdnet-lsd.service" | sudo tee /etc/systemd/system/$UNIDAD > /dev/null
sudo chmod 644 /etc/systemd/system/$UNIDAD
sudo systemctl daemon-reload
sudo systemctl enable $UNIDAD

echo ""
echo "==> Listo. Para arrancarlo ahora:"
echo "    sudo systemctl start $UNIDAD"
echo "    journalctl -u $UNIDAD -f       # o: tail -f $SCRIPT_DIR/motor.log"

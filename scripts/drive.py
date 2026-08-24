"""
Sube a Drive la deteccion recien exportada, apenas se genera -- mismo
principio que scripts/birdweather.py (evento por evento, no periodico).

Reemplaza el mecanismo anterior (sincronizar_detecciones.sh corriendo cada
5 minutos por crontab en LSD-Tector1.1, o el rclone copy masivo del arbol
completo al cierre de ventana en LSD-Tector2.0): en vez de barrer todo
By_Date/ cada vez, sube solo el archivo que se acaba de generar -- mas
liviano, y sin la demora de esperar al proximo ciclo periodico.
"""
import configparser
import os
import subprocess


def cargar_config_sincronizacion(path):
    """path: config/config_sincronizacion.txt (no el .ejemplo). Devuelve
    todo vacio/por-defecto si el archivo no existe todavia --
    subir_deteccion() ya sabe no hacer nada en ese caso, igual que con
    BirdWeather. Tambien viven aca REC_CARD/CHANNELS (tarjeta ALSA y
    canales del microfono): son especificos de cada dispositivo, igual
    que DRIVE_PATH, asi que van en el mismo archivo gitignored en vez de
    en config_deteccion.txt (que es compartido/versionado)."""
    if not os.path.isfile(path):
        return {
            'RCLONE_CONFIG': '', 'DRIVE_REMOTE': '', 'DRIVE_PATH': '', 'AUDIO_ROOT': '',
            'DRIVE_SUBCARPETA': 'Detecciones', 'REC_CARD': 'default', 'CHANNELS': 2,
        }
    parser = configparser.ConfigParser()
    with open(path) as f:
        contenido = '[DEFAULT]\n' + f.read()
    parser.read_string(contenido)
    c = parser['DEFAULT']
    return {
        'RCLONE_CONFIG': c.get('RCLONE_CONFIG', fallback='').strip(),
        'DRIVE_REMOTE': c.get('DRIVE_REMOTE', fallback='').strip(),
        'DRIVE_PATH': c.get('DRIVE_PATH', fallback='').strip(),
        'DRIVE_SUBCARPETA': c.get('DRIVE_SUBCARPETA', fallback='Detecciones').strip(),
        'REC_CARD': c.get('REC_CARD', fallback='default').strip(),
        'CHANNELS': c.getint('CHANNELS', fallback=2),
        'AUDIO_ROOT': c.get('AUDIO_ROOT', fallback='').strip(),
    }


def subir_deteccion(config_sync, ruta_local, carpeta_relativa, nombre_archivo, timeout=90):
    """ruta_local: archivo ya escrito en disco. carpeta_relativa y
    nombre_archivo: los que devuelve exportador.nombre_archivo(), ej
    carpeta_relativa='By_Date/2026-08-23/Rufous_Hornero'.

    Sube un solo archivo (copyto, no copy de todo el arbol) al mismo
    destino que ya usan cierre_amanecer.sh/cierre_atardecer.sh: esos
    scripts hacen `rclone copy .../By_Date/ gdrive:$DRIVE_PATH/Detecciones`
    -- con la barra final, rclone copia el CONTENIDO de By_Date, no la
    carpeta en si, asi que "By_Date" nunca llega a Drive (el destino real
    termina siendo gdrive:$DRIVE_PATH/Detecciones/<fecha>/<especie>/...).
    Aca se pisa el mismo comportamiento a mano, sacando el primer
    componente ("By_Date") de carpeta_relativa antes de armar el destino
    -- localmente el archivo si se guarda bajo By_Date/ (ver AUDIO_ROOT en
    motor.py), eso no cambia, es solo el destino remoto el que lo omite.
    Cualquier barrido periodico que siga corriendo por fuera encuentra el
    archivo ya arriba y no hace nada de mas (rclone copy es idempotente).

    Mismo timeout defensivo que el resto de las llamadas a rclone del
    proyecto (ver nota sobre la cuota compartida en el README de
    LSD-Tector1.1): si se satura, esta subida puntual se salta en vez de
    trabar el resto del loop de deteccion.

    Devuelve True si subio, False si no habia config o si fallo/dio
    timeout -- nunca lanza excepcion, una deteccion perdida de Drive no
    debe tirar abajo el resto del motor."""
    remote = config_sync.get('DRIVE_REMOTE') or ''
    drive_path = config_sync.get('DRIVE_PATH') or ''
    if not remote or not drive_path:
        return False

    partes = carpeta_relativa.split(os.sep)
    if partes and partes[0] == 'By_Date':
        partes = partes[1:]
    carpeta_relativa_remota = '/'.join(partes)

    subcarpeta = config_sync.get('DRIVE_SUBCARPETA') or 'Detecciones'
    destino = f'{remote}:{drive_path}/{subcarpeta}/{carpeta_relativa_remota}/{nombre_archivo}'
    env = os.environ.copy()
    if config_sync.get('RCLONE_CONFIG'):
        env['RCLONE_CONFIG'] = config_sync['RCLONE_CONFIG']

    try:
        resultado = subprocess.run(
            ['rclone', 'copyto', ruta_local, destino],
            timeout=timeout, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        return resultado.returncode == 0
    except subprocess.TimeoutExpired:
        return False

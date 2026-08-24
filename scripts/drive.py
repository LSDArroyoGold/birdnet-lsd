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
    todo vacio si el archivo no existe todavia -- subir_deteccion() ya
    sabe no hacer nada en ese caso, igual que con BirdWeather."""
    if not os.path.isfile(path):
        return {'RCLONE_CONFIG': '', 'DRIVE_REMOTE': '', 'DRIVE_PATH': '', 'AUDIO_ROOT': ''}
    parser = configparser.ConfigParser()
    with open(path) as f:
        contenido = '[DEFAULT]\n' + f.read()
    parser.read_string(contenido)
    c = parser['DEFAULT']
    return {
        'RCLONE_CONFIG': c.get('RCLONE_CONFIG', fallback='').strip(),
        'DRIVE_REMOTE': c.get('DRIVE_REMOTE', fallback='').strip(),
        'DRIVE_PATH': c.get('DRIVE_PATH', fallback='').strip(),
        'AUDIO_ROOT': c.get('AUDIO_ROOT', fallback='').strip(),
    }


def subir_deteccion(config_sync, ruta_local, carpeta_relativa, nombre_archivo, timeout=90):
    """ruta_local: archivo ya escrito en disco. carpeta_relativa y
    nombre_archivo: los que devuelve exportador.nombre_archivo(), ej
    carpeta_relativa='By_Date/2026-08-23/Rufous_Hornero'.

    Sube un solo archivo (copyto, no copy de todo el arbol) al mismo
    destino que ya usan cierre_amanecer.sh/cierre_atardecer.sh
    (gdrive:$DRIVE_PATH/Detecciones/By_Date/...) -- cualquier barrido
    periodico que siga corriendo por fuera encuentra el archivo ya
    arriba y no hace nada de mas (rclone copy es idempotente).

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

    destino = f'{remote}:{drive_path}/Detecciones/{carpeta_relativa}/{nombre_archivo}'
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

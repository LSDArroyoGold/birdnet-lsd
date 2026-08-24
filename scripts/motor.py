"""
Loop principal en vivo: arecord -> AcumuladorEventos -> Clasificador ->
audio_io (mp3 a disco) -> BirdWeather + Drive, los dos apenas se genera
cada deteccion.

Reemplaza tanto birdnet_recording.sh + birdnet_analysis.py de BirdNET-Pi
(grabacion + clasificacion ventana por ventana sin confirmacion) como el
mecanismo de sincronizacion periodico (sincronizar_detecciones.sh cada 5
minutos por crontab en LSD-Tector1.1, o el rclone copy del arbol completo
al cierre de ventana en LSD-Tector2.0): ahora el ciclo completo
detectar -> guardar -> avisar es un solo evento encadenado por deteccion,
sin esperar a ningun cron ni a que termine la ventana de grabacion.
"""
import configparser
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_io
import birdweather
import drive
import exportador
from clasificador import Clasificador
from detector import SR, AcumuladorEventos, cargar_config, cargar_config_birdweather

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cargar_config_captura(path):
    """Lee REC_CARD/CHANNELS/DURACION_BLOQUE_S del mismo
    config_deteccion.txt -- cargar_config() de detector.py ignora estas
    claves porque solo extrae las que ya conocia, asi que conviven sin
    problema en el mismo archivo."""
    parser = configparser.ConfigParser()
    with open(path) as f:
        contenido = '[DEFAULT]\n' + f.read()
    parser.read_string(contenido)
    c = parser['DEFAULT']
    return {
        'REC_CARD': c.get('REC_CARD', fallback='default').strip(),
        'CHANNELS': c.getint('CHANNELS', fallback=1),
        'DURACION_BLOQUE_S': c.getfloat('DURACION_BLOQUE_S', fallback=5.0),
    }


def leer_bloques_arecord(rec_card, channels, duracion_bloque_s):
    """Generador infinito de (bloque_mono_float32, timestamp_inicio_bloque),
    leidos en vivo desde ALSA via arecord -- mismo binario y mismos
    parametros de captura (S16_LE, 48kHz) que usa birdnet_recording.sh de
    BirdNET-Pi. Si arecord se cae (mic desconectado, error de ALSA), se
    reintenta solo despues de una pausa en vez de tirar abajo el motor
    entero -- un corte de audio no debe matar el proceso."""
    frame_bytes = 2 * channels  # S16_LE = 2 bytes por muestra por canal
    bloque_bytes = int(SR * duracion_bloque_s) * frame_bytes

    while True:
        cmd = ['arecord', '-f', 'S16_LE', '-c', str(channels), '-r', str(SR), '-t', 'raw']
        if rec_card and rec_card != 'default':
            cmd += ['-D', rec_card]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            while True:
                inicio_bloque = datetime.now()
                crudo = proc.stdout.read(bloque_bytes)
                if len(crudo) < bloque_bytes:
                    break  # arecord murio a mitad de bloque
                muestras = np.frombuffer(crudo, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    muestras = muestras.reshape(-1, channels).mean(axis=1)
                yield muestras, inicio_bloque
        finally:
            proc.kill()
            proc.wait()
        print('arecord se corto, reintentando en 3s...', file=sys.stderr, flush=True)
        time.sleep(3)


def procesar_evento(clasificador, evento, config_sync, config_bw):
    audio = evento['audio']
    timestamp_inicio = evento['timestamp_inicio']
    if timestamp_inicio is None or len(audio) < SR * 1.0:
        return  # evento sin timestamp real o demasiado corto para tener sentido

    ventanas = clasificador.analizar_ventanas_todas(audio)
    resultado = clasificador.decidir_confianza_racha(ventanas)
    if resultado is None:
        return

    nombre, carpeta_relativa = exportador.nombre_archivo(resultado, timestamp_inicio)
    ruta_local = os.path.join(config_sync['AUDIO_ROOT'], carpeta_relativa, nombre)
    audio_io.escribir_mp3(audio, SR, ruta_local)

    print(
        f"[{timestamp_inicio.isoformat()}] {resultado['especie_comun']} "
        f"({resultado['confianza']:.2f}, racha={resultado['racha_maxima']}) -> {nombre}",
        flush=True,
    )

    if resultado.get('confianza_baja'):
        # Racha de una sola ventana, sin corroboracion de vecinas -- mas
        # propensa a error (confirmado el 24/08 probando en campo: un
        # canto largo real partido en 2 eventos dio una especie
        # incorrecta con 96% de confianza en el fragmento de racha=1, y
        # la especie correcta recien en el fragmento con racha=4). No se
        # descarta la deteccion (sigue guardada local y en Drive como
        # evidencia), pero no se postea a BirdWeather como si fuera una
        # identificacion confirmada.
        print(f"BirdWeather: salteado por confianza_baja (racha=1) -- {resultado['especie_comun']}",
              flush=True)
    else:
        try:
            birdweather.enviar_deteccion(config_bw, resultado, timestamp_inicio, audio, SR)
        except Exception as e:
            print(f'BirdWeather fallo (sigue el loop): {e}', file=sys.stderr, flush=True)

    if not drive.subir_deteccion(config_sync, ruta_local, carpeta_relativa, nombre):
        print(f'Drive: no se pudo subir {nombre} (sigue el loop, sin reintentar)',
              file=sys.stderr, flush=True)


def main():
    ruta_config_deteccion = os.path.join(BASE_DIR, 'config/config_deteccion.txt')
    config_det = cargar_config(ruta_config_deteccion)
    config_cap = cargar_config_captura(ruta_config_deteccion)
    config_bw = cargar_config_birdweather(os.path.join(BASE_DIR, 'config/config_birdweather.txt'))
    config_sync = drive.cargar_config_sincronizacion(
        os.path.join(BASE_DIR, 'config/config_sincronizacion.txt'))

    clasificador = Clasificador(
        os.path.join(BASE_DIR, 'modelo/LSDTector_Classifier_v2.tflite'),
        os.path.join(BASE_DIR, 'modelo/LSDTector_Classifier_v2_Labels.txt'),
        config_det,
        podado_json=os.path.join(BASE_DIR, 'modelo/v10_podado.json'),
        bias_json=os.path.join(BASE_DIR, 'modelo/stock_regional_meta.json'),
    )
    acumulador = AcumuladorEventos(config_det, duracion_bloque_s=config_cap['DURACION_BLOQUE_S'])

    print(
        f"birdnet-lsd arrancando (bloque={config_cap['DURACION_BLOQUE_S']}s, "
        f"card={config_cap['REC_CARD']}, canales={config_cap['CHANNELS']})",
        flush=True,
    )

    for bloque, timestamp_bloque in leer_bloques_arecord(
        config_cap['REC_CARD'], config_cap['CHANNELS'], config_cap['DURACION_BLOQUE_S']
    ):
        acumulador.procesar_bloque(bloque, timestamp_bloque=timestamp_bloque)
        while acumulador.eventos_terminados:
            evento = acumulador.eventos_terminados.pop(0)
            try:
                procesar_evento(clasificador, evento, config_sync, config_bw)
            except Exception as e:
                print(f'Error procesando evento (sigue el loop): {e}', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()

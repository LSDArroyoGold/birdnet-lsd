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
from scipy.signal import filtfilt, iirnotch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_io
import birdweather
import drive
import exportador
from clasificador import Clasificador
from detector import SR, AcumuladorEventos, cargar_config, cargar_config_birdweather

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cargar_duracion_bloque(path):
    """Lee DURACION_BLOQUE_S de config_deteccion.txt -- cargar_config() de
    detector.py ignora esta clave porque solo extrae las que ya conocia,
    asi que conviven sin problema en el mismo archivo. REC_CARD/CHANNELS
    (especificos del microfono de cada dispositivo, no del algoritmo) se
    leen aparte, de config_sincronizacion.txt -- ver drive.py."""
    parser = configparser.ConfigParser()
    with open(path) as f:
        contenido = '[DEFAULT]\n' + f.read()
    parser.read_string(contenido)
    return parser['DEFAULT'].getfloat('DURACION_BLOQUE_S', fallback=5.0)


def cargar_config_filtro_50hz(path):
    """Lee FILTRO_50HZ_HABILITADO/FILTRO_50HZ_FRECUENCIAS de
    config_deteccion.txt. Confirmado el 24/08 con ruido real capturado del
    mic de tector2: pico angosto y fuerte en 50Hz (+25.4dB sobre las
    frecuencias vecinas, firma tipica de ruido de red electrica argentina)
    y su 3er armonico en 150Hz (+29.9dB) -- 60Hz (no armonico de 50Hz) no
    mostro nada, asi que es especificamente esto, no ruido grave en
    general."""
    parser = configparser.ConfigParser()
    with open(path) as f:
        contenido = '[DEFAULT]\n' + f.read()
    parser.read_string(contenido)
    c = parser['DEFAULT']
    habilitado = c.getboolean('FILTRO_50HZ_HABILITADO', fallback=False)
    frecuencias = [float(f) for f in c.get('FILTRO_50HZ_FRECUENCIAS', fallback='50,100,150').split(',')]
    return habilitado, frecuencias


def construir_filtro_50hz(frecuencias, sr=SR, q=30):
    """Un notch angosto (scipy.signal.iirnotch) por frecuencia, para
    aplicar en cascada -- Q=30 lo mantiene LOCALIZADO: medido el 24/08,
    el corte real en 50/100/150Hz fue de -16.6 a -19.6dB, mientras que en
    las frecuencias intermedias (75/125/175Hz) el corte fue de solo -2.6 a
    -5.7dB (residual normal de cualquier notch real, no un pasa-banda
    ancho que se coma 50-150Hz entero)."""
    return [iirnotch(f, q, sr) for f in frecuencias]


def aplicar_filtro_50hz(audio, filtros):
    for b, a in filtros:
        audio = filtfilt(b, a, audio)
    return audio


def leer_bloques_arecord(rec_card, channels, duracion_bloque_s, filtro_50hz=None):
    """Generador infinito de (bloque_mono_float32, timestamp_inicio_bloque),
    leidos en vivo desde ALSA via arecord -- mismo binario y mismos
    parametros de captura (S16_LE, 48kHz) que usa birdnet_recording.sh de
    BirdNET-Pi. Si arecord se cae (mic desconectado, error de ALSA), se
    reintenta solo despues de una pausa en vez de tirar abajo el motor
    entero -- un corte de audio no debe matar el proceso.

    filtro_50hz: lista de (b, a) de construir_filtro_50hz(), o None para
    no filtrar -- aplicado aca, lo mas cerca posible del mic, para que
    tanto el disparador/acumulador como el audio exportado vean el mismo
    audio ya limpio."""
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
                if filtro_50hz is not None:
                    muestras = aplicar_filtro_50hz(muestras, filtro_50hz)
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
    duracion_bloque_s = cargar_duracion_bloque(ruta_config_deteccion)
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
    acumulador = AcumuladorEventos(config_det, duracion_bloque_s=duracion_bloque_s)

    filtro_50hz_on, frecuencias_50hz = cargar_config_filtro_50hz(ruta_config_deteccion)
    filtro_50hz = construir_filtro_50hz(frecuencias_50hz) if filtro_50hz_on else None

    print(
        f"birdnet-lsd arrancando (bloque={duracion_bloque_s}s, "
        f"card={config_sync['REC_CARD']}, canales={config_sync['CHANNELS']}, "
        f"filtro_50hz={'on ' + str(frecuencias_50hz) if filtro_50hz_on else 'off'})",
        flush=True,
    )

    for bloque, timestamp_bloque in leer_bloques_arecord(
        config_sync['REC_CARD'], config_sync['CHANNELS'], duracion_bloque_s, filtro_50hz=filtro_50hz
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

"""
Loop principal en vivo: arecord -> AcumuladorEventos -> ClasificadorTectorNet
(Perch2 ONNX decide, BirdSet ONNX confirma) -> audio_io (mp3 a disco) ->
BirdWeather + Drive, los dos apenas se genera cada deteccion.

Reemplaza tanto birdnet_recording.sh + birdnet_analysis.py de BirdNET-Pi
(grabacion + clasificacion ventana por ventana sin confirmacion) como el
mecanismo de sincronizacion periodico (sincronizar_detecciones.sh cada 5
minutos por crontab en LSD-Tector1.1, o el rclone copy del arbol completo
al cierre de ventana en LSD-Tector2.0): el ciclo completo detectar ->
guardar -> avisar es un solo evento encadenado por deteccion, sin esperar
a ningun cron ni a que termine la ventana de grabacion.

Cambio de clasificador el 29/08/2026: el modelo BirdNET reentrenado
(tflite, clasificador.py -- ver tag "pre-tectornet" e historial de git
para el codigo viejo completo) se retira por completo, reemplazado por
TectorNet (Perch 2.0 ONNX como decisor + BirdSet EfficientNetB1 ONNX
como filtro de confirmacion cruzada, sin sesgo). Motivo: revision manual
encontro que el reentreno sobre-disparaba de forma estructural
(etiquetaba practicamente cualquier cosa como Hornero). Detalle completo
de la arquitectura nueva en clasificador_tectornet.py.

Cambio de arquitectura el mismo dia: la captura de audio (leer el pipe de
arecord) y la clasificacion de eventos ya NO corren en el mismo hilo.
Motivo, medido en campo el 29/08 sobre tector1 (Pi4B real): clasificar un
evento de 60s con TectorNet tardaba 99.74s -- mas lento que el propio
evento, y hasta un evento chico de 4.5s tardaba 5.78s. Con el loop viejo
(donde procesar_evento() bloqueaba el mismo hilo que lee arecord), esos
segundos de clasificacion dejaban de leer el pipe de arecord, que tiene
un buffer chico (tipicamente 64KB en Linux, se llena en fracciones de
segundo a la tasa de captura real, 48kHz/16-bit/2ch = ~192KB/s) -- al
llenarse, arecord se bloquea escribiendo y ALSA empieza a perder audio
real (overrun), justo mientras el sistema esta identificando una
deteccion. No era un problema con BirdNET tflite (mucho mas liviano),
pero es inaceptable con TectorNet. Ver hilo_clasificador() y
COLA_EVENTOS mas abajo.

Estos dos cambios (recorte de BirdSet a la racha de Perch2, paso_s=2.0,
DURACION_MAXIMA_EVENTO_S=20) redujeron el tiempo de clasificacion medido
en campo entre 1.6x y 3.6x segun duracion del evento (mayor beneficio en
eventos largos) -- probado localmente contra 150 archivos reales de
campo (0 crashes) antes de este despliegue, y confirmado en tector1
mismo la noche del 29/08 antes de reemplazar el clasificador en
produccion.
"""
import configparser
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
import huggingface_hub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_io
import birdweather
import drive
import exportador
from clasificador_tectornet import ClasificadorTectorNet, cargar_config_tectornet
from detector import SR, AcumuladorEventos, cargar_config, cargar_config_birdweather

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _detectar_log_sistema_real():
    """birdnet-lsd es un repo hermano independiente, compartido tal cual
    entre tector1 (LSD-Tector1.1) y tector2 (LSD-Tector2.0) -- pero cada
    uno tiene su PROPIA convencion de ruta para el log_sistema.txt que de
    verdad se sube a Drive en cada ventana:
      - tector2 (LSD-Tector2.0): anidado, /home/lsd/LSD-Tector2.0/log_sistema.txt
      - tector1 (LSD-Tector1.1): plano, /home/lsd/log_sistema.txt directo
    Bug real encontrado el 4/9/2026: la primera version de este archivo
    (mismo dia) hardcodeaba la ruta de tector2 -- arreglaba las alertas
    ahi pero las rompia en tector1 en silencio apenas bajara este mismo
    commit por git pull, exactamente la clase de bug de "arregla uno,
    rompe el otro" que se estaba arreglando. Se detecta por la presencia
    de la carpeta LSD-Tector2.0 (no existe en tector1) en vez de asumir
    una sola convencion."""
    if os.path.isdir("/home/lsd/LSD-Tector2.0"):
        return "/home/lsd/LSD-Tector2.0/log_sistema.txt"
    return "/home/lsd/log_sistema.txt"


_LOG_SISTEMA_REAL = _detectar_log_sistema_real()


def _alertar(mensaje):
    """Alerta que SI llega a Drive (a diferencia de un print a motor.log,
    que solo vive en el disco local del dispositivo, invisible sin SSH).
    Mismo formato de linea que ya escriben log_sistema.py (MSG) y
    chequeo_bateria.sh, para que sea un solo log consistente."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    try:
        with open(_LOG_SISTEMA_REAL, 'a') as f:
            f.write(f'[{timestamp}] {mensaje}\n')
    except Exception:
        pass  # best-effort -- si esto falla, ya se imprimio a stderr/motor.log igual

PERCH2_REPO = "justinchuby/Perch-onnx"
PERCH2_CHECKPOINT = "perch_v2_no_dft.onnx"

# Tope de eventos pendientes de clasificar antes de empezar a descartar
# los mas nuevos. No es "cuanto tarda en ponerse al dia" -- es una red de
# seguridad de memoria: si el clasificador queda sistematicamente mas
# lento que el ritmo real de eventos, esta cola creceria sin limite y
# podria agotar la RAM del dispositivo. Cada evento en cola pesa como
# maximo DURACION_MAXIMA_EVENTO_S (20s) de audio float32 a 48kHz mono,
# ~3.8MB -- con este tope, ~19MB en el peor caso, nada frente a los
# ~934MB que ya usan los modelos cargados (medido en tector1, Pi4B, el
# 29/08). Si se llega a este tope de forma sostenida es señal de que el
# clasificador no da abasto con la actividad real del sitio, no algo
# para simplemente subir el numero.
MAX_COLA_EVENTOS = 5


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


def procesar_evento(clasificador, evento, config_sync, config_bw, paso_ventana_s=1.0):
    audio = evento['audio']
    timestamp_inicio = evento['timestamp_inicio']
    if timestamp_inicio is None or len(audio) < SR * 1.0:
        return  # evento sin timestamp real o demasiado corto para tener sentido

    ventanas = clasificador.analizar_ventanas_todas(audio, paso_s=paso_ventana_s)
    resultado = clasificador.decidir_confianza_racha(ventanas, incluir_diagnostico=True)
    if resultado is None:
        print(f"[{timestamp_inicio.isoformat()}] disparador activo, evento sin ventanas "
              f"analizables (audio={len(audio)/SR:.1f}s)", flush=True)
        return
    if not resultado['detectado']:
        print(f"[{timestamp_inicio.isoformat()}] disparador activo, sin clasificacion confiable "
              f"(mejor candidato: {resultado['especie_comun']} {resultado['confianza']:.2f} "
              f"< umbral {clasificador.confidence}, audio={len(audio)/SR:.1f}s)", flush=True)
        return

    nombre, carpeta_relativa = exportador.nombre_archivo(resultado, timestamp_inicio)
    ruta_local = os.path.join(config_sync['AUDIO_ROOT'], carpeta_relativa, nombre)
    audio_io.escribir_mp3(audio, SR, ruta_local)

    if resultado['confirmado_por_birdset']:
        estado_birdset = 'confirma'
    else:
        estado_birdset = f"NO confirma (dijo {resultado['especie_birdset']})"
    print(
        f"[{timestamp_inicio.isoformat()}] {resultado['especie_comun']} "
        f"({resultado['confianza']:.2f}, racha={resultado['racha_maxima']}, "
        f"BirdSet={estado_birdset}) -> {nombre}",
        flush=True,
    )

    if resultado.get('confianza_baja'):
        # BirdSet (sin sesgo, corriendo sobre el mismo tramo de audio que
        # gano la racha de Perch2) no confirmo la especie que propuso
        # Perch2 -- ver clasificador_tectornet.py para la nota completa
        # sobre por que el umbral propio de Perch2 solo no alcanza como
        # filtro de calidad. No se descarta la deteccion (sigue guardada
        # local y en Drive como evidencia), pero no se postea a
        # BirdWeather como si fuera una identificacion confirmada.
        print(f"BirdWeather: salteado por confianza_baja (BirdSet no confirmo) -- {resultado['especie_comun']}",
              flush=True)
    else:
        try:
            birdweather.enviar_deteccion(config_bw, resultado, timestamp_inicio, audio, SR)
        except Exception as e:
            print(f'BirdWeather fallo (sigue el loop): {e}', file=sys.stderr, flush=True)
            _alertar(f'ALERTA: BirdWeather fallo para {resultado["especie_comun"]} -- {e}')

    if not drive.subir_deteccion(config_sync, ruta_local, carpeta_relativa, nombre):
        print(f'Drive: no se pudo subir {nombre} (sigue el loop, sin reintentar)',
              file=sys.stderr, flush=True)
        _alertar(f'ALERTA: no se pudo subir a Drive {nombre} (sin reintentar)')


def hilo_clasificador(cola_eventos, clasificador, config_sync, config_bw, paso_ventana_s):
    """Corre en un hilo aparte del que lee arecord (ver docstring del
    modulo). Consume eventos ya delimitados y hace todo el trabajo
    pesado -- clasificar, guardar mp3, BirdWeather, Drive -- sin
    bloquear nunca la captura de audio en vivo. onnxruntime libera el
    GIL durante sess.run() (ejecucion en C++), y las operaciones pesadas
    de librosa/numpy tambien lo liberan durante las llamadas a BLAS, asi
    que un hilo (no un proceso aparte) alcanza sin agregar la complejidad
    de pasar arrays de audio entre procesos."""
    while True:
        evento = cola_eventos.get()
        try:
            procesar_evento(clasificador, evento, config_sync, config_bw, paso_ventana_s=paso_ventana_s)
        except Exception as e:
            print(f'Error procesando evento (sigue el loop): {e}', file=sys.stderr, flush=True)
        finally:
            cola_eventos.task_done()


def main():
    ruta_config_deteccion = os.path.join(BASE_DIR, 'config/config_deteccion.txt')
    config_det = cargar_config(ruta_config_deteccion)
    config_det.update(cargar_config_tectornet(ruta_config_deteccion))
    duracion_bloque_s = cargar_duracion_bloque(ruta_config_deteccion)
    config_bw = cargar_config_birdweather(os.path.join(BASE_DIR, 'config/config_birdweather.txt'))
    config_sync = drive.cargar_config_sincronizacion(
        os.path.join(BASE_DIR, 'config/config_sincronizacion.txt'))

    print('Descargando/verificando cache de Perch 2.0 ONNX (HuggingFace, primera vez tarda)...', flush=True)
    perch_onnx_path = huggingface_hub.hf_hub_download(repo_id=PERCH2_REPO, filename=PERCH2_CHECKPOINT)

    clasificador = ClasificadorTectorNet(
        perch_onnx_path=perch_onnx_path,
        perch_labels_csv=os.path.join(BASE_DIR, 'modelo/perch2_labels.csv'),
        birdset_onnx_path=os.path.join(BASE_DIR, 'modelo/birdset_efficientnetb1.onnx'),
        birdset_config_json=os.path.join(BASE_DIR, 'modelo/birdset_efficientnetb1_config.json'),
        ebird_taxonomy_json=os.path.join(BASE_DIR, 'modelo/eBird_taxonomy_codes_2024E.json'),
        config=config_det,
        bias_json=os.path.join(BASE_DIR, 'config/delta_b_reforzado_v2.json'),
        escala=config_det['ESCALA'],
    )
    acumulador = AcumuladorEventos(config_det, duracion_bloque_s=duracion_bloque_s)

    cola_eventos = queue.Queue(maxsize=MAX_COLA_EVENTOS)
    hilo = threading.Thread(
        target=hilo_clasificador,
        args=(cola_eventos, clasificador, config_sync, config_bw, config_det['PASO_VENTANA_S']),
        daemon=True,
    )
    hilo.start()

    print(
        f"birdnet-lsd (TectorNet) arrancando (bloque={duracion_bloque_s}s, escala={config_det['ESCALA']}, "
        f"paso_ventana={config_det['PASO_VENTANA_S']}s, card={config_sync['REC_CARD']}, "
        f"canales={config_sync['CHANNELS']})",
        flush=True,
    )

    for bloque, timestamp_bloque in leer_bloques_arecord(
        config_sync['REC_CARD'], config_sync['CHANNELS'], duracion_bloque_s
    ):
        acumulador.procesar_bloque(bloque, timestamp_bloque=timestamp_bloque)
        while acumulador.eventos_terminados:
            evento = acumulador.eventos_terminados.pop(0)
            try:
                cola_eventos.put_nowait(evento)
            except queue.Full:
                # Ver MAX_COLA_EVENTOS: el clasificador no da abasto con el
                # ritmo real de eventos. Se descarta el evento MAS NUEVO (no
                # el mas viejo) para no interrumpir al hilo clasificador a
                # mitad de un evento que ya empezo a procesar -- prioriza
                # terminar lo que ya esta en curso antes que sumar mas cola.
                print(
                    f'ALERTA: cola de clasificacion llena ({MAX_COLA_EVENTOS} eventos pendientes), '
                    f'se descarta un evento nuevo -- el clasificador esta mas lento que el ritmo real',
                    file=sys.stderr, flush=True,
                )


if __name__ == '__main__':
    main()

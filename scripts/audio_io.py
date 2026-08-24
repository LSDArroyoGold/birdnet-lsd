"""
Escritura del audio de un evento a disco, como mp3 -- unica pieza de
E/S de audio que le faltaba al motor (el resto ya trabaja en memoria con
arrays de numpy). Usa ffmpeg (ya presente en el dispositivo -- lo usa
birdnet_recording.sh de BirdNET-Pi para el modo RTSP) en vez de sox/lame,
para no sumar una dependencia nueva al sistema.

320kbps, sin ningun lowpass -- mismo criterio aplicado el 23/08 al
pipeline de BirdNET-Pi via configurar_birdnet.sh (ahi el problema era el
lowpass por defecto de LAME al pasar por sox; aca, al llamar a ffmpeg
directo, ese problema ni existe).
"""
import os
import subprocess

import numpy as np


def escribir_mp3(audio, sr, ruta_destino):
    """audio: array float32 mono en rango [-1, 1] (aplanado si viniera de
    una fuente multicanal -- ver leer_bloques_arecord() en motor.py, que
    ya hace ese downmix antes de que el audio llegue a ningun otro lado
    del pipeline). Crea las carpetas intermedias si hace falta."""
    os.makedirs(os.path.dirname(ruta_destino), exist_ok=True)
    audio_clip = np.clip(audio, -1.0, 1.0).astype(np.float32)

    proc = subprocess.run(
        ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
         '-f', 'f32le', '-ar', str(sr), '-ac', '1', '-i', '-',
         '-codec:a', 'libmp3lame', '-b:a', '320k', ruta_destino],
        input=audio_clip.tobytes(),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f'ffmpeg fallo escribiendo {ruta_destino}: '
            f'{proc.stderr.decode(errors="replace")}'
        )

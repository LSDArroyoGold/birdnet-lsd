"""
Compara las 4 formas de decision (pico, mayoria, confianza acumulada,
racha) sobre el mismo set de 15 pares Hornero/Kestrel reconstruidos,
usando la metrica (Hornero - Kestrel) / (Hornero + Kestrel): +1 = ideal
(todos Hornero, cero Kestrel), -1 = todo lo contrario, 0 = empate.
"""
import sys, os, glob, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from datetime import datetime
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

from detector import cargar_config, AcumuladorEventos, SR
from clasificador import Clasificador

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_deteccion.txt')
config = cargar_config(CONFIG_PATH)

MODEL = os.path.expanduser('~/Desktop/Tector/LSDTector-BirdNET-custom-v1/model/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite')
LABELS = os.path.expanduser('~/Desktop/Tector/LSDTector-BirdNET-custom-v1/model/BirdNET_GLOBAL_6K_V2.4_Model_FP16_Labels.txt')
clasificador = Clasificador(MODEL, LABELS, config)

BASE = os.path.expanduser('~/Desktop/Tector/Datasets_prueba/BirdNET_Detecciones')
patron = re.compile(r'^(.+)-(\d+)-(\d{4}-\d{2}-\d{2})-birdnet-(\d{2})[_:](\d{2})[_:](\d{2})\.mp3$')

def parsear(path):
    m = patron.match(os.path.basename(path))
    if not m:
        return None
    especie, conf, fecha, hh, mm, ss = m.groups()
    dt = datetime.strptime(f'{fecha} {hh}:{mm}:{ss}', '%Y-%m-%d %H:%M:%S')
    return {'path': path, 'especie': especie, 'conf': int(conf), 'dt': dt}

detecciones = []
for especie in ['American_Kestrel', 'Rufous_Hornero']:
    for f in glob.glob(os.path.join(BASE, '*', especie, '*.mp3')):
        r = parsear(f)
        if r:
            detecciones.append(r)
detecciones.sort(key=lambda r: r['dt'])

pares = []
for i in range(len(detecciones) - 1):
    a, b = detecciones[i], detecciones[i + 1]
    if a['especie'] == b['especie']:
        continue
    gap = (b['dt'] - a['dt']).total_seconds()
    if 0 < gap <= 10:
        pares.append((a, b))

print(f'{len(pares)} pares\n')

FRAME, HOP = 1024, 256
sos = butter(4, [config['TRIGGER_BANDA_MIN_HZ'], config['TRIGGER_BANDA_MAX_HZ']], btype='band', fs=SR, output='sos')

def recortar_a_senal_activa(y, margen_s=0.15, percentil_piso=20, margen_db=8):
    filtrado = sosfiltfilt(sos, y)
    rms = librosa.feature.rms(y=filtrado, frame_length=FRAME, hop_length=HOP)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
    piso = np.percentile(rms_db, percentil_piso)
    activo = rms_db > (piso + margen_db)
    if not activo.any():
        return y
    tiempos = librosa.frames_to_time(np.arange(len(activo)), sr=SR, hop_length=HOP)
    ini = max(0, tiempos[np.argmax(activo)] - margen_s)
    fin = min(len(y) / SR, tiempos[len(activo) - 1 - np.argmax(activo[::-1])] + (HOP / SR) + margen_s)
    return y[int(ini * SR):int(fin * SR)]

def es(r, nombre_cientifico):
    return bool(r and r.get('especie') and nombre_cientifico in r['especie'] and r.get('detectado'))

METODOS = ['decidir_pico', 'decidir_mayoria', 'decidir_confianza_acumulada', 'decidir_racha']

# baseline: cada pieza recortada, sola, decidida con 'racha' (la mas
# conservadora) -- referencia fija, la misma para las 4 comparaciones
base_hornero = base_kestrel = 0

resultados_por_metodo = {m: {'hornero': 0, 'kestrel': 0, 'ninguna': 0} for m in METODOS}

for a, b in pares:
    ya, _ = librosa.load(a['path'], sr=SR, mono=True)
    yb, _ = librosa.load(b['path'], sr=SR, mono=True)
    ya_rec = recortar_a_senal_activa(ya)
    yb_rec = recortar_a_senal_activa(yb)

    r_a = clasificador.clasificar_evento(ya_rec)
    r_b = clasificador.clasificar_evento(yb_rec)
    base_hornero += any(es(r, 'Furnarius rufus') for r in (r_a, r_b))
    base_kestrel += any(es(r, 'Falco sparverius') for r in (r_a, r_b))

    acumulador = AcumuladorEventos(config, duracion_bloque_s=len(ya_rec) / SR)
    acumulador.procesar_bloque(ya_rec)
    acumulador.procesar_bloque(yb_rec)
    acumulador.finalizar()

    if not acumulador.eventos_terminados:
        ventanas = []
    else:
        evento = max(acumulador.eventos_terminados, key=len)
        ventanas = clasificador.analizar_ventanas(evento, paso_s=1.0)

    for metodo in METODOS:
        r = getattr(clasificador, metodo)(ventanas) if ventanas else None
        if es(r, 'Furnarius rufus'):
            resultados_por_metodo[metodo]['hornero'] += 1
        elif es(r, 'Falco sparverius'):
            resultados_por_metodo[metodo]['kestrel'] += 1
        else:
            resultados_por_metodo[metodo]['ninguna'] += 1

def ratio(h, k):
    return (h - k) / (h + k) if (h + k) > 0 else 0.0

print('===== Baseline (piezas sueltas, como hoy) =====')
print(f'  Hornero={base_hornero}/{len(pares)}  Kestrel={base_kestrel}/{len(pares)}  ratio=(H-K)/(H+K)={ratio(base_hornero, base_kestrel):+.3f}\n')

print('===== Motor nuevo, por criterio de decision =====')
for metodo, r in resultados_por_metodo.items():
    print(f"  {metodo:<30s} Hornero={r['hornero']:2d}  Kestrel={r['kestrel']:2d}  ninguna={r['ninguna']:2d}  "
          f"ratio={ratio(r['hornero'], r['kestrel']):+.3f}")

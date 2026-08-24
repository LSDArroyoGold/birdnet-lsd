"""Clasificacion sobre un evento ya delimitado por el AcumuladorEventos.

Expone las 4 formas de decision que se probaron el 22/08 para comparar
entre si con la misma metrica: pico de confianza mas alto en cualquier
ventana, mayoria de ventanas ganadas, confianza acumulada (suma), y racha
de ventanas consecutivas mas larga. Todas parten del mismo calculo de
ventanas (analizar_ventanas), una sola pasada de inferencia por ventana,
reutilizable para las 4."""
import os, json
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
from tensorflow import lite as tflite

SR = 48000
VENTANA_S = 3.0


def sens_from_conf(sensitivity):
    return max(0.5, min(1.0 - (sensitivity - 1.0), 1.5))


class Clasificador:
    def __init__(self, model_path, labels_path, config, podado_json=None, bias_json=None):
        """podado_json: ruta opcional a un archivo tipo v10_podado.json.
        Cuando se pasa, la neurona reentrenada (bare) de cada especie
        listada en "neuronas_suprimidas" queda invisible para la decision
        -- su score siempre se fuerza a 0, y la neurona ORIGINAL de esa
        especie (que ya esta en el mismo modelo Append, sin tocar) es la
        unica que puede disparar esa especie. No requiere modificar el
        .tflite, es una mascara aplicada en cada corrida.

        bias_json: ruta opcional a un archivo tipo stock_regional_meta.json
        (el mismo sesgo regional ya validado en LSDTector-BirdNET-custom-v1,
        por nombre cientifico). Se suma al logit crudo antes del sigmoide,
        para CUALQUIER neurona que represente esa especie -- tanto la bare
        reentrenada como la original combinada, porque el sesgo es sobre la
        especie en si (frecuencia real en la region), no sobre una neurona
        puntual. El modelo de v10 no lo trae horneado (es un .tflite
        distinto al de custom-v1), asi que se aplica aca en el codigo."""
        with open(labels_path) as f:
            self.labels = [l.strip() for l in f]

        # Tabla cientifico -> comun, construida de las propias labels: las
        # entradas ORIGINALES (no reentrenadas) vienen en formato
        # "Cientifico_Comun"; las reentrenadas (bare) son solo el nombre
        # cientifico, sin comun. Para poder nombrar archivos de forma
        # consistente con el resto del sistema (que usa nombre comun en
        # todos lados) hace falta resolver el comun tambien para esas.
        self.sci_a_comun = {}
        for l in self.labels:
            if '_' in l:
                sci, com = l.split('_', 1)
                self.sci_a_comun[sci.strip()] = com.strip()

        self.interp = tflite.Interpreter(model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()
        self.out = self.interp.get_output_details()
        self.escala = sens_from_conf(config['SENSITIVITY'])
        self.confidence = config['CONFIDENCE']

        self.indices_suprimidos = set()
        if podado_json:
            with open(podado_json) as f:
                podado = json.load(f)
            self.indices_suprimidos = {
                d['idx_bare_suprimido'] for d in podado['neuronas_suprimidas'].values()
                if d.get('idx_bare_suprimido') is not None
            }

        self.bias_vector = np.zeros(len(self.labels), dtype=np.float32)
        if bias_json:
            with open(bias_json) as f:
                bias_meta = json.load(f)
            delta_b = bias_meta['delta_b']
            for i, l in enumerate(self.labels):
                sci = l.split('_', 1)[0].strip()  # bare o combinada, el cientifico va primero en ambas
                if sci in delta_b:
                    self.bias_vector[i] = delta_b[sci]

    def _score_ventana(self, chunk):
        x = np.zeros(int(SR * VENTANA_S), dtype=np.float32)
        n = min(len(chunk), len(x))
        x[:n] = chunk[:n]
        self.interp.set_tensor(self.inp[0]['index'], x[np.newaxis, :])
        self.interp.invoke()
        logits = self.interp.get_tensor(self.out[0]['index'])[0]
        scores = 1 / (1 + np.exp(-self.escala * (logits + self.bias_vector)))
        if self.indices_suprimidos:
            for idx in self.indices_suprimidos:
                scores[idx] = 0.0
        return scores

    def analizar_ventanas(self, audio, paso_s=1.0):
        """Devuelve la lista (idx_especie_o_None, confianza) de cada
        ventana de 3s a lo largo de todo el audio -- solo el top1 de cada
        ventana, y solo si cruza CONFIDENCE (si no, None). Base comun para
        pico/mayoria/confianza_acumulada/racha (las 4 formas comparadas el
        22/08). OJO: filtra por umbral ANTES de agrupar, por eso no sirve
        de base para decidir_confianza_racha (ver analizar_ventanas_todas)."""
        win = int(VENTANA_S * SR)
        paso = int(paso_s * SR)
        resultado = []
        for ini in range(0, max(1, len(audio) - int(SR * 1.0)), paso):
            chunk = audio[ini:ini + win]
            scores = self._score_ventana(chunk)
            idx_top = int(np.argmax(scores))
            if scores[idx_top] >= self.confidence:
                resultado.append((idx_top, float(scores[idx_top])))
            else:
                resultado.append((None, 0.0))
        return resultado

    def nombre_comun_de(self, label):
        """Nombre comun para cualquier label, sea original ("Cientifico_Comun")
        o reentrenada bare ("Cientifico" solo) -- resuelve siempre al mismo
        nombre comun para una especie, sin importar que neurona la disparo."""
        if '_' in label:
            return label.split('_', 1)[1].strip()
        return self.sci_a_comun.get(label.strip(), label.strip())

    def nombre_comun_seguro(self, label):
        """Version 'safe' para nombre de archivo/carpeta, mismo criterio que
        common_name_safe de BirdNET-Pi: espacios -> guion bajo, sin comillas."""
        return self.nombre_comun_de(label).replace("'", "").replace(' ', '_')

    def analizar_ventanas_todas(self, audio, paso_s=1.0):
        """Igual que analizar_ventanas, pero SIN filtrar por CONFIDENCE --
        devuelve el top1 real de cada ventana (idx, confianza) siempre,
        nunca None. Pensada para decidir_confianza_racha: el umbral se
        aplica reciEn al final, sobre la confianza ya agregada de todo el
        evento, no ventana por ventana de entrada."""
        win = int(VENTANA_S * SR)
        paso = int(paso_s * SR)
        resultado = []
        for ini in range(0, max(1, len(audio) - int(SR * 1.0)), paso):
            chunk = audio[ini:ini + win]
            scores = self._score_ventana(chunk)
            idx_top = int(np.argmax(scores))
            resultado.append((idx_top, float(scores[idx_top])))
        return resultado

    def _agrupar_por_especie(self, ventanas):
        cuentas = {}
        for idx, c in ventanas:
            if idx is not None:
                cuentas.setdefault(idx, []).append(c)
        return cuentas

    def decidir_pico(self, ventanas):
        """La ventana individual de mayor confianza en todo el evento,
        sin importar cuantas veces se repita esa especie."""
        candidatas = [(idx, c) for idx, c in ventanas if idx is not None]
        if not candidatas:
            return None
        idx, c = max(candidatas, key=lambda x: x[1])
        return {'especie': self.labels[idx], 'confianza': c, 'detectado': True}

    def decidir_mayoria(self, ventanas):
        """La especie que gana mas ventanas (conteo), empate por confianza media."""
        cuentas = self._agrupar_por_especie(ventanas)
        if not cuentas:
            return None
        idx = max(cuentas, key=lambda i: (len(cuentas[i]), np.mean(cuentas[i])))
        return {'especie': self.labels[idx], 'confianza': float(np.mean(cuentas[idx])), 'detectado': True}

    def decidir_confianza_acumulada(self, ventanas):
        """La especie con mayor SUMA de confianza entre las ventanas que gano."""
        cuentas = self._agrupar_por_especie(ventanas)
        if not cuentas:
            return None
        idx = max(cuentas, key=lambda i: sum(cuentas[i]))
        return {'especie': self.labels[idx], 'confianza': float(np.mean(cuentas[idx])), 'detectado': True}

    def decidir_racha(self, ventanas, min_racha=2):
        """La racha mas larga de ventanas CONSECUTIVAS ganadas por la
        misma especie (no cuenta si son ventanas salteadas)."""
        rachas = {}
        i = 0
        while i < len(ventanas):
            idx, c = ventanas[i]
            if idx is None:
                i += 1
                continue
            j = i
            confs = []
            while j < len(ventanas) and ventanas[j][0] == idx:
                confs.append(ventanas[j][1])
                j += 1
            largo = j - i
            if largo > rachas.get(idx, (0, []))[0]:
                rachas[idx] = (largo, confs)
            i = j
        candidatos = {idx: d for idx, d in rachas.items() if d[0] >= min_racha}
        if not candidatos:
            return None
        idx = max(candidatos, key=lambda i: (candidatos[i][0], np.mean(candidatos[i][1])))
        largo, confs = candidatos[idx]
        return {'especie': self.labels[idx], 'confianza': float(np.mean(confs)), 'detectado': True}

    def decidir_confianza_racha(self, ventanas):
        """Hibrido definido el 23/08, flujo corregido el mismo dia:

        1. NO se filtra por CONFIDENCE ventana por ventana de entrada (por
           eso usa analizar_ventanas_todas, no analizar_ventanas) -- un
           canto real de una sola ventana (menos de ~6s) no puede quedar
           afuera de la carrera solo por no tener con quien formar racha.
        2. Para cada racha de ventanas CONSECUTIVAS ganadas por la misma
           especie, se calcula un PROMEDIO PONDERADO por posicion dentro
           de la racha (1a ventana pesa 1, 2a pesa 2, 3a pesa 3, ...):
              confianza_racha = sum(conf_i * i) / sum(i)
           Una racha de 1 sola ventana da exactamente la confianza cruda
           de esa ventana (sin bonus ni penalidad) -- por eso un canto
           corto real sigue pudiendo confirmarse solo, con su propia
           confianza, ni mas ni menos.
        3. Se compara la confianza ya agregada de cada especie (su mejor
           racha) contra CONFIDENCE reciEn en este paso final -- no antes.
           Una racha larga sostenida en confianzas moderadas puede terminar
           con un promedio ponderado mas alto que un pico aislado corto,
           porque las ultimas ventanas de la racha (las que mas pesan) son
           las que consolidan la tendencia.

        Devuelve tambien 'confianza_baja': True cuando la racha ganadora es
        chica Y ademas no es la totalidad del evento (hubo otras ventanas
        que no fueron parte de ella). Investigacion real (24/08): notas de
        ave individuales pueden durar apenas 8-400ms -- mucho menos que
        VENTANA_S=3.0s -- asi que un canto genuinamente corto puede
        producir de forma legitima una racha de 1-2 ventanas nada mas, sin
        que eso sea sintoma de mala deteccion. Lo que si es sospechoso es
        una racha chica ganando en un evento MAS LARGO, donde el resto de
        las ventanas fueron para otras especies o quedaron sin clasificar
        con claridad -- eso es evidencia debil (un "blip" aislado), no un
        canto corto real. Por eso NO alcanza con mirar el largo de racha
        solo: racha_maxima == total_ventanas (todo el evento coincidio, sin
        excepcion) nunca es confianza_baja aunque sea chica; racha_maxima
        chica Y menor al total si lo es. No descarta la deteccion, solo la
        marca para quien quiera loguearlo distinto (ej. no mandarlo a
        BirdWeather pero si al log local)."""
        rachas_por_especie = {}  # idx -> lista de (largo, confianza_ponderada) de cada racha vista
        i = 0
        while i < len(ventanas):
            idx, _ = ventanas[i]
            j = i
            while j < len(ventanas) and ventanas[j][0] == idx:
                j += 1
            largo = j - i
            confs = [ventanas[k][1] for k in range(i, j)]
            pesos = list(range(1, largo + 1))
            confianza_ponderada = sum(c * p for c, p in zip(confs, pesos)) / sum(pesos)
            rachas_por_especie.setdefault(idx, []).append((largo, confianza_ponderada))
            i = j

        # de cada especie, su MEJOR racha (por confianza ponderada, no por largo)
        mejor_por_especie = {
            idx: max(rachas, key=lambda rc: rc[1])
            for idx, rachas in rachas_por_especie.items()
        }

        if not mejor_por_especie:
            return None
        idx_top = max(mejor_por_especie, key=lambda i: mejor_por_especie[i][1])
        largo, confianza = mejor_por_especie[idx_top]

        if confianza < self.confidence:
            return None

        return {
            'especie': self.labels[idx_top],
            'especie_comun': self.nombre_comun_de(self.labels[idx_top]),
            'confianza': float(confianza),
            'racha_maxima': largo,
            'total_ventanas': len(ventanas),
            'confianza_baja': largo <= 2 and largo < len(ventanas),
            'detectado': True,
        }

    # Metodo de conveniencia usado en las pruebas: confianza acumulada + racha.
    def clasificar_evento(self, audio, paso_s=1.0):
        ventanas = self.analizar_ventanas_todas(audio, paso_s=paso_s)
        r = self.decidir_confianza_racha(ventanas)
        if r is None:
            return {'especie': None, 'confianza': 0.0, 'detectado': False}
        return r

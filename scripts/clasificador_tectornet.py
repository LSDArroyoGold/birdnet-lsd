"""Clasificacion sobre un evento ya delimitado por AcumuladorEventos
(detector.py, reusado tal cual de birdnet-lsd -- es agnostico al modelo).

Arquitectura TectorNet, decidida y validada el 28/08/2026 tras descartar
BirdNET reentrenado (sobre-disparaba: etiquetaba cualquier cosa como
Hornero) y Perch v1 vía TensorFlow completo (pesaba demasiado, >1.5GB de
pico de RAM con el resto del pipeline):

  DECISOR -- Perch 2.0 ONNX (`perch_v2_no_dft.onnx`, HuggingFace
  `justinchuby/Perch-onnx`, Apache 2.0): decide la especie. Cargado
  DIRECTO con onnxruntime.InferenceSession, sin bioacoustics_model_zoo
  (bmz importa opensoundscape, que importa torch/torchaudio SIEMPRE,
  aunque nunca se use ningun modelo de torch -- confirmado empiricamente
  el 28/08). El grafo ONNX "no_dft" ya incluye el DFT/espectrograma
  adentro (aplanado a operaciones ONNX estandar en vez de un op nativo de
  DFT) -- el input es la onda cruda (32kHz, 5.0s = 160000 muestras),
  normalizada a pico=0.25, sin ningun preprocesamiento en Python. Labels:
  nombre cientifico directo (14795 clases, columna 'inat2024_fsd50k' de
  sammlapp/Perch_v2_headless/labels.csv). Sesgo regional
  (delta_b_reforzado_v2.json, ALPHA=0.7) sumado al logit ANTES del
  sigmoide, por nombre cientifico directo (sin tabla de codigos eBird,
  a diferencia de Perch v1). ESCALA=0.15 multiplicando el exponente del
  sigmoide (recalibrado el 28/08 -- ver nota importante mas abajo).

  FILTRO -- BirdSet EfficientNetB1 ONNX (exportado localmente el 28/08
  con `optimum-cli export onnx`, ver modelo/birdset_efficientnetb1.onnx):
  confirma o no la especie que propuso Perch, SIN sesgo regional (es el
  chequeo independiente). Preprocesamiento propio en preprocess_birdset.py
  (reimplementado en librosa puro, sin torch, validado el 28/08 contra el
  preprocesador torch/opensoundscape original: 19/20 top-1 coinciden en
  el pipeline end-to-end). Labels: codigos eBird de 6 letras (mismo
  formato que ya usa el resto del proyecto), resueltos a nombre
  cientifico via eBird_taxonomy_codes_2024E.json. Corre SOLO sobre el
  tramo de audio de la racha ganadora de Perch2, no sobre el evento
  completo (corregido el 29/08, ver nota en analizar_ventanas_todas).

  DECISION CONJUNTA: Perch decide. BirdSet confirma SOLO si su propio
  top-1 (sin sesgo), calculado sobre el MISMO tramo de audio que gano la
  racha de Perch, coincide EXACTO con la especie que propuso Perch.
  Confirmado -> se postea a BirdWeather como de costumbre. Descartado ->
  se guarda local (evidencia) pero NO se postea -- exactamente el mismo
  mecanismo que ya usaba el motor real para 'confianza_baja' (racha de 1
  ventana sin corroboracion), reusado aca para la corroboracion cruzada
  entre modelos en vez de la corroboracion entre ventanas vecinas.

NOTA IMPORTANTE sobre el flag 'detectado' de Perch: con 14795 clases
candidatas + sesgo aditivo, el valor crudo (logit+bias) de la clase
GANADORA de una ventana es, en la practica, PRACTICAMENTE SIEMPRE
positivo (estadistica de extremos sobre miles de clases + el sesgo
regional que solo suma, nunca resta lo suficiente) -- confirmado el 28/08
sobre 2887 ventanas reales de campo: el 100% de las ventanas ganadoras
tenian valor crudo > 0, minimo 6.33. Como sigmoid(escala*x) > 0.5 para
CUALQUIER escala>0 mientras x>0, esto significa que 'detectado' en Perch
es estructuralmente casi siempre True, sin importar que escala se elija
-- NO es un bug de calibracion que la escala pueda arreglar, es una
propiedad matematica de este tipo de arquitectura (argmax sobre miles de
clases + sesgo aditivo). Bajar la escala solo reduce el NUMERO de
confianza reportado (util para no ver todo pegado a 0.999), pero no
convierte 'detectado' en una señal de calidad confiable por si sola.
EL FILTRADO REAL DE CALIDAD LO HACE BIRDSET (la confirmacion cruzada), NO
el umbral propio de Perch -- ver 'confianza_baja'/'confirmado_por_birdset'
en el resultado de decidir_confianza_racha().
"""
import csv
import json
import os

import numpy as np
import onnxruntime as ort
import librosa

import preprocess_birdset

SR_MOTOR = 48000        # SR interno del motor (AcumuladorEventos, detector.py) -- NO se toca
SR_MODELOS = 32000      # sample rate nativo de Perch2 y BirdSet
VENTANA_PERCH_S = 5.0   # ventana nativa de Perch2 (fija, el grafo ONNX espera exactamente esta duracion)
VENTANA_BIRDSET_S = 5.0  # ventana nativa de BirdSet


def _resample_evento(audio_48k):
    """Evento delimitado por AcumuladorEventos, a SR_MOTOR (48kHz) ->
    SR_MODELOS (32kHz). Un solo resample, reusado para Perch2 y BirdSet."""
    if len(audio_48k) == 0:
        return audio_48k.astype(np.float32)
    return librosa.resample(
        np.asarray(audio_48k, dtype=np.float32), orig_sr=SR_MOTOR, target_sr=SR_MODELOS,
    )


def _normalizar_pico(audio, pico=0.25):
    """Mismo preprocesamiento que bmz.Perch2ONNX() aplica antes de
    Perch2 (Audio.normalize(peak_level=0.25)) -- reimplementado sin
    depender de opensoundscape."""
    maximo = np.abs(audio).max()
    if maximo > 0:
        return audio * (pico / maximo)
    return audio


class ClasificadorTectorNet:
    """Interfaz deliberadamente compatible con Clasificador (clasificador.py
    de birdnet-lsd) en los metodos que usa motor.py: .confidence,
    .analizar_ventanas_todas(audio, paso_s), .decidir_confianza_racha(
    ventanas, incluir_diagnostico), .nombre_comun_de(label). No hace
    falta ningun cambio en motor.py mas alla de importar esta clase en
    vez de la original y ajustar las rutas de carga (ver scripts/motor.py
    de este repo)."""

    def __init__(self, perch_onnx_path, perch_labels_csv, birdset_onnx_path, birdset_config_json,
                 ebird_taxonomy_json, config, bias_json=None, escala=1.0):
        """
        perch_onnx_path: ruta local a perch_v2_no_dft.onnx (descargado en
            runtime desde HuggingFace por instalar.sh/el llamador -- ver
            scripts/motor.py).
        perch_labels_csv: modelo/perch2_labels.csv (columna 'inat2024_fsd50k').
        birdset_onnx_path: modelo/birdset_efficientnetb1.onnx (committeado).
        birdset_config_json: modelo/birdset_efficientnetb1_config.json
            (trae el id2label -> lista de codigos eBird, mismo orden que
            los logits de salida del modelo).
        ebird_taxonomy_json: modelo/eBird_taxonomy_codes_2024E.json
            (bidireccional "Sci Name_Common Name" <-> codigo eBird de 6
            letras -- resuelve los codigos de BirdSet a nombre cientifico,
            y da nombre comun para ambos modelos cuando la especie esta en
            el catalogo de 6522 especies de BirdNET; para especies fuera
            de ese catalogo -- Perch2/BirdSet cubren bastantes mas -- el
            nombre comun cae de vuelta al nombre cientifico).
        config: dict de detector.cargar_config() (CONFIDENCE, etc.) MAS
            'ESCALA' (ver cargar_config_tectornet() mas abajo).
        bias_json: ruta a delta_b_reforzado_v2.json, o None para correr a
            Perch2 sin sesgo regional (crudo). Solo se aplica a Perch2 --
            BirdSet nunca lleva sesgo, es el chequeo independiente.
        escala: multiplica el exponente del sigmoide de Perch2 (ver nota
            del docstring del modulo sobre por que esto NO controla
            'detectado' de forma confiable).
        """
        self.confidence = config['CONFIDENCE']
        self.escala = escala

        # --- Perch2 (decisor) ---
        self.sess_perch = ort.InferenceSession(perch_onnx_path, providers=["CPUExecutionProvider"])
        self._perch_input_name = self.sess_perch.get_inputs()[0].name
        with open(perch_labels_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # header (inat2024_fsd50k)
            self.labels = [row[0] for row in reader]  # nombre cientifico directo, indexado por idx

        self.bias_vector = np.zeros(len(self.labels), dtype=np.float32)
        if bias_json:
            with open(bias_json, encoding="utf-8") as f:
                delta_b = json.load(f)["delta_b"]
            for i, sci in enumerate(self.labels):
                if sci in delta_b:
                    self.bias_vector[i] = delta_b[sci]

        # --- BirdSet (filtro) ---
        self.sess_birdset = ort.InferenceSession(birdset_onnx_path, providers=["CPUExecutionProvider"])
        self._birdset_input_name = self.sess_birdset.get_inputs()[0].name
        with open(birdset_config_json, encoding="utf-8") as f:
            birdset_cfg = json.load(f)
        id2label = birdset_cfg["id2label"]
        self.birdset_codigos = [id2label[str(i)] for i in range(len(id2label))]

        # --- taxonomia (comun para los dos modelos) ---
        with open(ebird_taxonomy_json, encoding="utf-8") as f:
            self._taxonomia = json.load(f)
        self._sci_a_comun = {}
        for k, v in self._taxonomia.items():
            if isinstance(k, str) and "_" in k:
                sci, comun = k.split("_", 1)
                self._sci_a_comun[sci.strip()] = comun.strip()

        # se completa en analizar_ventanas_todas(), lo consume decidir_confianza_racha()
        self._ultimo_birdset_sci = None

    # ------------------------------------------------------------------
    # Resolucion de nombres (misma interfaz que Clasificador original)
    # ------------------------------------------------------------------
    def _codigo_ebird_a_sci(self, codigo):
        entrada = self._taxonomia.get(codigo)
        if entrada is None or "_" not in entrada:
            return None
        return entrada.split("_", 1)[0].strip()

    def nombre_comun_de(self, label):
        """label: nombre cientifico (Perch2) o 'Cientifico_Comun' (formato
        sintetico que arma decidir_confianza_racha para compatibilidad con
        birdweather.py, ver mas abajo). Si la especie no esta en el
        catalogo de 6522 de BirdNET (Perch2/BirdSet cubren mas), devuelve
        el nombre cientifico tal cual -- documentado, no es un error."""
        if "_" in label:
            return label.split("_", 1)[1].strip()
        return self._sci_a_comun.get(label.strip(), label.strip())

    def nombre_comun_seguro(self, label):
        return self.nombre_comun_de(label).replace("'", "").replace(" ", "_")

    # ------------------------------------------------------------------
    # Perch2: ventaneo + sesgo + escala (el "decisor")
    # ------------------------------------------------------------------
    def _score_ventana_perch(self, chunk_32k):
        win = int(VENTANA_PERCH_S * SR_MODELOS)
        x = np.zeros(win, dtype=np.float32)
        n = min(len(chunk_32k), win)
        x[:n] = chunk_32k[:n]
        x = _normalizar_pico(x)
        logits = self.sess_perch.run(["label"], {self._perch_input_name: x[np.newaxis, :]})[0][0]
        scores = 1.0 / (1.0 + np.exp(-self.escala * (logits + self.bias_vector)))
        return scores

    def analizar_ventanas_todas(self, audio, paso_s=1.0):
        """audio: evento delimitado por AcumuladorEventos, a SR_MOTOR
        (48kHz). Devuelve la lista (idx_especie, confianza) de cada
        ventana de Perch2 a lo largo de todo el evento -- mismo formato
        que Clasificador.analizar_ventanas_todas original (BirdNET), para
        que decidir_confianza_racha (la misma logica de racha ponderada)
        funcione identico.

        De paso, corre BirdSet -- pero SOLO sobre el tramo de audio donde
        Perch2 encontro su racha ganadora, no sobre el evento completo.
        Corregido el 29/08/2026: la version anterior hacia que BirdSet
        reescaneara TODO el evento por su cuenta, buscando su propia mejor
        racha de forma independiente -- eso comparaba dos busquedas
        independientes que podian corresponder a momentos distintos del
        mismo audio (ej. dos aves distintas en el mismo evento largo), no
        una confirmacion real de la misma evidencia que uso Perch2 para
        decidir. Ademas era carisimo: BirdSet terminaba procesando todas
        las ventanas del evento (hasta 59 en un evento de 60s) en vez de
        solo las pocas de la racha ganadora (tipicamente 1-4), que era el
        cuello de botella de rendimiento mas grande medido en campo el
        29/08 (tector1, Pi4B real: 99.74s para clasificar un evento de
        60s, mas lento que el propio evento)."""
        audio_32k = _resample_evento(audio)

        win = int(VENTANA_PERCH_S * SR_MODELOS)
        paso = int(paso_s * SR_MODELOS)
        resultado = []
        for ini in range(0, max(1, len(audio_32k) - int(SR_MODELOS * 1.0)), paso):
            chunk = audio_32k[ini:ini + win]
            scores = self._score_ventana_perch(chunk)
            idx_top = int(np.argmax(scores))
            resultado.append((idx_top, float(scores[idx_top])))

        ganadora = _racha_ganadora(resultado)
        if ganadora is None:
            self._ultimo_birdset_sci = None
        else:
            _idx_top, _conf, _largo, i_inicio, i_fin = ganadora
            ini_muestra = i_inicio * paso
            fin_muestra = min(len(audio_32k), i_fin * paso + win)  # +win: cubre toda la ultima ventana de la racha, no solo su inicio
            recorte = audio_32k[ini_muestra:fin_muestra]
            self._ultimo_birdset_sci = self._decidir_birdset(recorte, paso_s)
        return resultado

    # ------------------------------------------------------------------
    # BirdSet: ventaneo + racha, SIN sesgo (el "filtro")
    # ------------------------------------------------------------------
    def _decidir_birdset(self, audio_32k, paso_s=1.0):
        win = int(VENTANA_BIRDSET_S * SR_MODELOS)
        paso = int(paso_s * SR_MODELOS)
        ventanas = []
        for ini in range(0, max(1, len(audio_32k) - int(SR_MODELOS * 1.0)), paso):
            chunk = audio_32k[ini:ini + win]
            x = np.zeros(win, dtype=np.float32)
            n = min(len(chunk), win)
            x[:n] = chunk[:n]
            pixel_values = preprocess_birdset.audio_a_pixel_values(x)
            logits = self.sess_birdset.run(["logits"], {self._birdset_input_name: pixel_values})[0][0]
            scores = 1.0 / (1.0 + np.exp(-logits))
            idx_top = int(np.argmax(scores))
            ventanas.append((idx_top, float(scores[idx_top])))

        if not ventanas:
            return None
        mejor = _racha_ganadora(ventanas)
        if mejor is None:
            return None
        idx_top, _confianza, _largo, _i_inicio, _i_fin = mejor
        return self._codigo_ebird_a_sci(self.birdset_codigos[idx_top])

    # ------------------------------------------------------------------
    # Decision final: racha ponderada de Perch2 (identica al motor
    # original) + confirmacion cruzada de BirdSet
    # ------------------------------------------------------------------
    def decidir_confianza_racha(self, ventanas, incluir_diagnostico=False):
        """Copia funcional de Clasificador.decidir_confianza_racha
        (clasificador.py de birdnet-lsd): rachas de ventanas consecutivas
        ganadas por la misma clase, promedio ponderado por posicion
        dentro de la racha, compara la mejor racha de cada especie recien
        al final contra CONFIDENCE. Unico agregado: la confirmacion
        cruzada de BirdSet (self._ultimo_birdset_sci, calculado en la
        misma llamada a analizar_ventanas_todas que produjo `ventanas`,
        ya sobre el recorte de la racha ganadora -- ver _racha_ganadora()).
        """
        ganadora = _racha_ganadora(ventanas)
        if ganadora is None:
            return None
        idx_top, confianza, largo, _i_inicio, _i_fin = ganadora

        if confianza < self.confidence:
            if incluir_diagnostico:
                return {
                    "especie_comun": self.nombre_comun_de(self.labels[idx_top]),
                    "confianza": float(confianza),
                    "detectado": False,
                }
            return None

        sci = self.labels[idx_top]
        comun = self.nombre_comun_de(sci)
        especie_combinada = f"{sci}_{comun}"  # formato "Cientifico_Comun", compatible con birdweather.py

        confirmado = (
            self._ultimo_birdset_sci is not None and self._ultimo_birdset_sci == sci
        )

        return {
            "especie": especie_combinada,
            "especie_comun": comun,
            "especie_birdset": self._ultimo_birdset_sci,
            "confirmado_por_birdset": confirmado,
            "confianza": float(confianza),
            "racha_maxima": largo,
            "total_ventanas": len(ventanas),
            # reusa el mismo campo que ya consumia motor.py para "guardar
            # local, no postear a BirdWeather" (antes: racha=1 sin
            # corroboracion de vecinas; ahora: BirdSet no confirmo) -- ver
            # nota del docstring del modulo sobre por que 'detectado' de
            # Perch2 NO es un filtro de calidad confiable por si solo.
            "confianza_baja": not confirmado,
            "detectado": True,
        }

    def clasificar_evento(self, audio, paso_s=1.0):
        """Metodo de conveniencia (igual que en el Clasificador original)."""
        ventanas = self.analizar_ventanas_todas(audio, paso_s=paso_s)
        r = self.decidir_confianza_racha(ventanas)
        if r is None:
            return {"especie": None, "confianza": 0.0, "detectado": False}
        return r


def _racha_ganadora(ventanas):
    """Encuentra, entre las rachas de ventanas consecutivas ganadas por la
    misma clase, la de mayor confianza ponderada por posicion dentro de
    la racha (misma logica que ya usaba decidir_confianza_racha).
    Ademas de la especie y la confianza, devuelve el RANGO de indices
    [i_inicio, i_fin) de esa racha dentro de `ventanas` -- lo necesita
    analizar_ventanas_todas() para saber que tramo de audio recortarle a
    BirdSet (ver comentario ahi), no solo cual especie gano.

    Unico punto de calculo de 'cual es la racha ganadora' en el modulo --
    tanto decidir_confianza_racha() (decision final de Perch2 a postear)
    como analizar_ventanas_todas() (que tramo pasarle a BirdSet) llaman a
    esta misma funcion, para no mantener la logica duplicada en 3 lugares
    (asi estaba antes del 29/08: decidir_confianza_racha y
    _mejor_racha_generico tenian el mismo bucle copiado dos veces)."""
    rachas_por_especie = {}
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
        rachas_por_especie.setdefault(idx, []).append((largo, confianza_ponderada, i, j))
        i = j

    mejor_por_especie = {
        idx: max(rachas, key=lambda rc: rc[1])
        for idx, rachas in rachas_por_especie.items()
    }
    if not mejor_por_especie:
        return None
    idx_top = max(mejor_por_especie, key=lambda i: mejor_por_especie[i][1])
    largo, confianza, i_inicio, i_fin = mejor_por_especie[idx_top]
    return idx_top, confianza, largo, i_inicio, i_fin


def cargar_config_tectornet(path):
    """Como detector.cargar_config() (que solo conoce las claves del
    motor original -- CONFIDENCE, SENSITIVITY, TRIGGER_*, etc. -- y ya se
    reusa tal cual, sin tocar), pero agrega las claves nuevas de
    TectorNet (ESCALA) que ese modulo no conoce. SENSITIVITY sigue
    presente en config_deteccion.txt por compatibilidad de formato con
    birdnet-lsd, pero ClasificadorTectorNet no la usa (era especifica del
    escalado de BirdNET/Clasificador original) -- se ignora."""
    import configparser
    parser = configparser.ConfigParser()
    with open(path) as f:
        contenido = "[DEFAULT]\n" + f.read()
    parser.read_string(contenido)
    c = parser["DEFAULT"]
    return {
        "ESCALA": c.getfloat("ESCALA", fallback=1.0),
        # Paso entre ventanas consecutivas (Perch2 decide sobre todo el
        # evento con este paso; BirdSet lo reusa sobre el recorte de la
        # racha ganadora). Subido de 1.0 a 2.0 el 29/08 -- Diego valido
        # con pruebas propias que el sistema anda igual con este paso,
        # y reduce a la mitad el numero de ventanas que Perch2 tiene que
        # escanear por evento (impacto directo en el tiempo de
        # clasificacion, ver actualizar_birdnet_lsd.sh/README para el
        # contexto completo de por que esto importaba).
        "PASO_VENTANA_S": c.getfloat("PASO_VENTANA_S", fallback=2.0),
    }

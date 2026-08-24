"""
Motor de deteccion propio: en vez de clasificar cada ventana fija de 3s de
forma independiente (como hace BirdNET-Pi hoy, sin ninguna confirmacion
entre ventanas vecinas -- confirmado leyendo su codigo real el 22/08),
espera a juntar el evento acustico completo (con buffer rodante que cruza
limites de bloque de grabacion) y clasifica una sola vez sobre el audio
entero. Resuelve al mismo tiempo el problema de clasificacion con
fragmentos cortos y el de entregar el audio de campo completo.

Este modulo es puro / offline-testable: no depende de arecord/ffmpeg ni de
ningun servicio -- recibe bloques de audio como arrays de numpy, en el
mismo orden en que llegarian en produccion. La integracion con la
grabacion real (systemd, watch de archivos) se hace despues, una vez
validada la logica.
"""
import configparser
import os
from datetime import timedelta
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

SR = 48000


def cargar_config(path):
    parser = configparser.ConfigParser()
    with open(path) as f:
        contenido = '[DEFAULT]\n' + f.read()
    parser.read_string(contenido)
    c = parser['DEFAULT']
    return {
        'CONFIDENCE': c.getfloat('CONFIDENCE'),
        'SENSITIVITY': c.getfloat('SENSITIVITY'),
        'TRIGGER_BANDA_MIN_HZ': c.getfloat('TRIGGER_BANDA_MIN_HZ'),
        'TRIGGER_BANDA_MAX_HZ': c.getfloat('TRIGGER_BANDA_MAX_HZ'),
        'TRIGGER_PERCENTIL_PISO': c.getfloat('TRIGGER_PERCENTIL_PISO'),
        'TRIGGER_MARGEN_DB': c.getfloat('TRIGGER_MARGEN_DB'),
        'SILENCIO_FIN_EVENTO_S': c.getfloat('SILENCIO_FIN_EVENTO_S'),
        'MARGEN_FINAL_PCT': c.getfloat('MARGEN_FINAL_PCT'),
        'DURACION_MAXIMA_EVENTO_S': c.getfloat('DURACION_MAXIMA_EVENTO_S'),
        'BLOQUES_BUFFER_ANTERIORES': c.getint('BLOQUES_BUFFER_ANTERIORES'),
        'ALPHA_PISO_EMA': c.getfloat('ALPHA_PISO_EMA'),
        'CANAL_GRAVE_HABILITADO': c.getboolean('CANAL_GRAVE_HABILITADO', fallback=False),
        'CANAL_GRAVE_MIN_HZ': c.getfloat('CANAL_GRAVE_MIN_HZ', fallback=60.0),
        'CANAL_GRAVE_MAX_HZ': c.getfloat('CANAL_GRAVE_MAX_HZ', fallback=300.0),
    }


def cargar_config_birdweather(path):
    """path: config/config_birdweather.txt (no config_birdweather.txt.ejemplo,
    ese es solo la plantilla). Devuelve BIRDWEATHER_ID vacio si el archivo
    no existe todavia -- birdweather.enviar_deteccion() ya sabe no hacer
    nada en ese caso, igual que BirdNET-Pi cuando no esta configurado."""
    if not os.path.isfile(path):
        return {'BIRDWEATHER_ID': '', 'LATITUDE': '', 'LONGITUDE': ''}
    parser = configparser.ConfigParser()
    with open(path) as f:
        contenido = '[DEFAULT]\n' + f.read()
    parser.read_string(contenido)
    c = parser['DEFAULT']
    return {
        'BIRDWEATHER_ID': c.get('BIRDWEATHER_ID', fallback='').strip(),
        'LATITUDE': c.get('LATITUDE', fallback='').strip(),
        'LONGITUDE': c.get('LONGITUDE', fallback='').strip(),
    }


class DetectorActividad:
    """Chequeo liviano (sin red neuronal) de si hay actividad acustica en
    un tramo de audio, usando la banda y el umbral relativo al piso de
    ruido propio ya validados empiricamente el 22/08."""

    def __init__(self, config, frame=1024, hop=256):
        self.frame = frame
        self.hop = hop
        self.sos = butter(
            4,
            [config['TRIGGER_BANDA_MIN_HZ'], config['TRIGGER_BANDA_MAX_HZ']],
            btype='band', fs=SR, output='sos',
        )
        self.percentil_piso = config['TRIGGER_PERCENTIL_PISO']
        self.margen_db = config['TRIGGER_MARGEN_DB']

        # Canal grave OPCIONAL (default apagado): segundo filtro pasa-banda
        # independiente, mas su propio piso de ruido, para especies con
        # vocalizaciones por debajo de la banda principal (ej. Anser anser,
        # Jacana jacana, Cinclodes fuscus: mediana espectral ~90-100Hz,
        # medido el 24/08 sobre las fallas persistentes del barrido nocturno
        # -- ver INFORME de esa sesion). Se combina con la banda principal
        # por OR (actividad si CUALQUIERA de las dos ve algo), nunca la
        # reemplaza ni le cambia el comportamiento cuando esta apagado --
        # sigue siendo exactamente el detector validado el 22-24/08.
        # Apagado por defecto a proposito: la banda grave tipica (60-300Hz)
        # se superpone con la fundamental de la voz humana (~85-255Hz) y
        # con ruido de manipulacion/viento/trafico -- en un dispositivo de
        # campo con gente cerca en ciertos horarios, mas disparos espurios
        # cuestan CPU/bateria de mas aunque el clasificador los termine
        # descartando. Activar solo si el sitio de despliegue realmente
        # tiene esas especies.
        self.canal_grave_habilitado = config['CANAL_GRAVE_HABILITADO']
        if self.canal_grave_habilitado:
            self.sos_grave = butter(
                4,
                [config['CANAL_GRAVE_MIN_HZ'], config['CANAL_GRAVE_MAX_HZ']],
                btype='band', fs=SR, output='sos',
            )

    def _rms_db(self, audio, sos):
        filtrado = sosfiltfilt(sos, audio)
        rms = librosa.feature.rms(y=filtrado, frame_length=self.frame, hop_length=self.hop)[0]
        return 20 * np.log10(np.maximum(rms, 1e-10))

    def mascara_actividad(self, audio, piso_db=None, piso_grave_db=None):
        """Devuelve (mascara_booleana_por_frame, tiempos_de_cada_frame).

        piso_db/piso_grave_db: piso de ruido fijo, en dB, para usar en vez
        de calcularlo con el percentil sobre `audio` (uno por canal).
        Fundamental para juzgar el CIERRE de un evento ya largo: si se
        recalcula el percentil sobre el propio evento acumulado (que puede
        ser ruidoso de punta a punta, ej. manipulacion del equipo durante
        una prueba), el piso se recalibra contra ese mismo ruido y el
        evento nunca se ve "en silencio" relativo a si mismo -- encontrado
        el 23-24/08 probando en campo (tres detecciones seguidas cortadas
        por el tope de DURACION_MAXIMA_EVENTO_S en vez de por silencio
        real). Con piso fijo (ver AcumuladorEventos.piso_referencia_db,
        calculado UNA vez del contexto anterior al disparo) el cierre se
        compara siempre contra "como sonaba antes de que arrancara esto",
        no contra el propio contenido del evento.

        Si el canal grave esta habilitado, la actividad es la UNION
        (OR) de lo que ve cada canal -- ver comentario en __init__."""
        rms_db = self._rms_db(audio, self.sos)
        piso = piso_db if piso_db is not None else np.percentile(rms_db, self.percentil_piso)
        activo = rms_db > (piso + self.margen_db)

        if self.canal_grave_habilitado:
            rms_db_grave = self._rms_db(audio, self.sos_grave)
            piso_grave = piso_grave_db if piso_grave_db is not None else np.percentile(rms_db_grave, self.percentil_piso)
            activo = activo | (rms_db_grave > (piso_grave + self.margen_db))

        tiempos = librosa.frames_to_time(np.arange(len(activo)), sr=SR, hop_length=self.hop)
        return activo, tiempos

    def piso_de(self, audio, grave=False):
        """Piso de ruido (percentil TRIGGER_PERCENTIL_PISO) de un tramo de
        audio, sin decidir actividad -- para congelar una referencia fija
        ANTES de que arranque un evento (ver AcumuladorEventos).
        grave=True usa el filtro del canal grave (solo tiene sentido si
        canal_grave_habilitado)."""
        sos = self.sos_grave if grave else self.sos
        rms_db = self._rms_db(audio, sos)
        return float(np.percentile(rms_db, self.percentil_piso))

    def hay_actividad(self, audio):
        mascara, _ = self.mascara_actividad(audio)
        return bool(mascara.any())


class AcumuladorEventos:
    """Recibe bloques de audio en orden (como llegarian de la grabacion
    real), mantiene un buffer rodante de los bloques anteriores, detecta
    cuando arranca un evento acustico, lo va acumulando -- cruzando
    bloques si hace falta -- y lo cierra cuando hay silencio sostenido o
    se llega al tope de seguridad. No clasifica nada; solo delimita audio."""

    def __init__(self, config, duracion_bloque_s):
        self.config = config
        self.detector = DetectorActividad(config)
        self.duracion_bloque_s = duracion_bloque_s
        self.buffer_bloques = []  # ultimos N bloques crudos, mas viejo primero
        self.n_buffer = config['BLOQUES_BUFFER_ANTERIORES']

        self.evento_audio = None       # None = no hay evento en curso
        self.timestamp_inicio_evento = None  # datetime real del inicio del evento en curso
        self.piso_referencia_db = None  # piso de ruido congelado del contexto anterior al disparo
        self.piso_referencia_grave_db = None  # idem, canal grave (solo si esta habilitado)
        self.muestras_desde_ultima_actividad = 0
        self.eventos_terminados = []   # lista de {'audio':..., 'timestamp_inicio':...}, uno por evento

    def _silencio_maximo_muestras(self):
        return int(self.config['SILENCIO_FIN_EVENTO_S'] * SR)

    def _maximo_evento_muestras(self):
        return int(self.config['DURACION_MAXIMA_EVENTO_S'] * SR)

    def procesar_bloque(self, bloque, timestamp_bloque=None):
        """bloque: array de audio de ~duracion_bloque_s segundos, en el
        orden real en que se grabo (bloques contiguos, sin salto).
        timestamp_bloque: datetime real del INICIO de este bloque (hora de
        pared) -- opcional; sin el, los eventos quedan sin timestamp (util
        para pruebas offline, pero en produccion hace falta para poder
        nombrar el archivo final)."""
        if self.evento_audio is None:
            self._buscar_disparo(bloque, timestamp_bloque)
        else:
            self._continuar_evento(bloque)

        self.buffer_bloques.append(bloque)
        if len(self.buffer_bloques) > self.n_buffer:
            self.buffer_bloques.pop(0)

    def _buscar_disparo(self, bloque, timestamp_bloque):
        mascara, tiempos = self.detector.mascara_actividad(bloque)
        if not mascara.any():
            return

        primer_frame_activo = int(np.argmax(mascara))
        inicio_s = tiempos[primer_frame_activo]

        # Reconstruir el contexto previo real desde el buffer rodante, no
        # solo desde donde arranca este bloque -- el canto puede haber
        # empezado un poco antes de que el disparador lo confirmara.
        contexto_previo = np.concatenate(self.buffer_bloques) if self.buffer_bloques else np.array([], dtype=np.float32)
        inicio_absoluto_muestra = len(contexto_previo) + int(inicio_s * SR)

        # Piso de referencia para todo este evento: del contexto ANTERIOR
        # al disparo (buffer rodante), o si todavia no hay buffer, del
        # tramo del propio bloque previo al onset -- se congela aca y no
        # se vuelve a tocar mientras dure el evento (ver mascara_actividad).
        tramo_referencia = contexto_previo if len(contexto_previo) >= SR * 0.5 else bloque[:max(1, int(inicio_s * SR))]
        hay_tramo = len(tramo_referencia) >= SR * 0.2
        self.piso_referencia_db = self.detector.piso_de(tramo_referencia) if hay_tramo else None
        if self.detector.canal_grave_habilitado:
            self.piso_referencia_grave_db = self.detector.piso_de(tramo_referencia, grave=True) if hay_tramo else None

        audio_total_disponible = np.concatenate([contexto_previo, bloque]) if len(contexto_previo) else bloque
        self.evento_audio = audio_total_disponible[max(0, inicio_absoluto_muestra):]
        self.muestras_desde_ultima_actividad = 0
        # el disparo se detecta siempre dentro del bloque ACTUAL (mascara_actividad
        # corre solo sobre `bloque`, no sobre el contexto previo), asi que la hora
        # real del inicio es la hora del bloque actual + el offset encontrado ahi.
        if timestamp_bloque is not None:
            self.timestamp_inicio_evento = timestamp_bloque + timedelta(seconds=inicio_s)
        else:
            self.timestamp_inicio_evento = None
        self._chequear_fin_o_continuar(bloque_actual_ya_incluido=True)

    def _continuar_evento(self, bloque):
        self.evento_audio = np.concatenate([self.evento_audio, bloque])

        # Promedio movil (EMA) del piso de referencia: se mezcla con el
        # piso propio de ESTE bloque nuevo, no con el evento acumulado
        # completo (eso reintroduciria el bug original -- ver
        # mascara_actividad). Deja que el piso se adapte a un ambiente
        # que sube y se sostiene (viento, trafico) en pocos bloques, sin
        # ser sensible a un canto real: piso_de() ya usa un percentil bajo
        # (TRIGGER_PERCENTIL_PISO), robusto a picos breves dentro del
        # bloque. Validado el 24/08 con barrido de alpha: 0.8 recupera en
        # ~3 bloques (~15s) ante un ambiente que sube y se mantiene, sin
        # cortar de mas ningun canto real probado (pausas internas cortas
        # bien por debajo de SILENCIO_FIN_EVENTO_S).
        alpha = self.config['ALPHA_PISO_EMA']
        if self.piso_referencia_db is not None:
            piso_bloque = self.detector.piso_de(bloque)
            self.piso_referencia_db = alpha * self.piso_referencia_db + (1 - alpha) * piso_bloque
        if self.detector.canal_grave_habilitado and self.piso_referencia_grave_db is not None:
            piso_bloque_grave = self.detector.piso_de(bloque, grave=True)
            self.piso_referencia_grave_db = alpha * self.piso_referencia_grave_db + (1 - alpha) * piso_bloque_grave

        self._chequear_fin_o_continuar(bloque_actual_ya_incluido=True)

    def _chequear_fin_o_continuar(self, bloque_actual_ya_incluido):
        mascara, tiempos = self.detector.mascara_actividad(
            self.evento_audio, piso_db=self.piso_referencia_db, piso_grave_db=self.piso_referencia_grave_db)

        if mascara.any():
            ultimo_frame_activo = len(mascara) - 1 - int(np.argmax(mascara[::-1]))
            fin_actividad_s = tiempos[ultimo_frame_activo] + (self.detector.hop / SR)
            silencio_actual_muestras = len(self.evento_audio) - int(fin_actividad_s * SR)
        else:
            silencio_actual_muestras = len(self.evento_audio)

        supera_tope = len(self.evento_audio) >= self._maximo_evento_muestras()
        supera_silencio = silencio_actual_muestras >= self._silencio_maximo_muestras()

        if supera_tope or supera_silencio:
            self._cerrar_evento(recortar_silencio_final=supera_silencio)

    def _cerrar_evento(self, recortar_silencio_final):
        audio_final = self.evento_audio
        if recortar_silencio_final:
            mascara, tiempos = self.detector.mascara_actividad(
                audio_final, piso_db=self.piso_referencia_db, piso_grave_db=self.piso_referencia_grave_db)
            if mascara.any():
                ultimo_frame_activo = len(mascara) - 1 - int(np.argmax(mascara[::-1]))
                fin_s = tiempos[ultimo_frame_activo] + (self.detector.hop / SR)
                margen_s = self.config['SILENCIO_FIN_EVENTO_S'] * (self.config['MARGEN_FINAL_PCT'] / 100.0)
                fin_muestra = min(len(audio_final), int((fin_s + margen_s) * SR))
                audio_final = audio_final[:fin_muestra]
        else:
            # Cierre por tope de seguridad: procesar_bloque() solo chequea
            # el tope DESPUES de pegar un bloque entero, asi que puede
            # sobrepasarlo hasta en un bloque completo (ej. 33.9s con
            # DURACION_MAXIMA_EVENTO_S=30) -- recortar con precision al
            # tope exacto en vez de dejar pasar ese sobrante.
            audio_final = audio_final[:self._maximo_evento_muestras()]
        self.eventos_terminados.append({
            'audio': audio_final,
            'timestamp_inicio': self.timestamp_inicio_evento,
        })
        self.evento_audio = None
        self.timestamp_inicio_evento = None
        self.piso_referencia_db = None
        self.piso_referencia_grave_db = None

    def finalizar(self):
        """Llamar al terminar el stream (o al apagar el dispositivo) para
        no perder un evento que haya quedado abierto sin llegar a cerrarse
        por silencio ni por tope."""
        if self.evento_audio is not None:
            self._cerrar_evento(recortar_silencio_final=False)

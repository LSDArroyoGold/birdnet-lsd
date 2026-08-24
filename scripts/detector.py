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

    def mascara_actividad(self, audio):
        """Devuelve (mascara_booleana_por_frame, tiempos_de_cada_frame)."""
        filtrado = sosfiltfilt(self.sos, audio)
        rms = librosa.feature.rms(y=filtrado, frame_length=self.frame, hop_length=self.hop)[0]
        rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
        piso = np.percentile(rms_db, self.percentil_piso)
        activo = rms_db > (piso + self.margen_db)
        tiempos = librosa.frames_to_time(np.arange(len(activo)), sr=SR, hop_length=self.hop)
        return activo, tiempos

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
        self.muestras_desde_ultima_actividad = 0
        self.eventos_terminados = []   # lista de arrays de audio, uno por evento

    def _silencio_maximo_muestras(self):
        return int(self.config['SILENCIO_FIN_EVENTO_S'] * SR)

    def _maximo_evento_muestras(self):
        return int(self.config['DURACION_MAXIMA_EVENTO_S'] * SR)

    def procesar_bloque(self, bloque):
        """bloque: array de audio de ~duracion_bloque_s segundos, en el
        orden real en que se grabo (bloques contiguos, sin salto)."""
        if self.evento_audio is None:
            self._buscar_disparo(bloque)
        else:
            self._continuar_evento(bloque)

        self.buffer_bloques.append(bloque)
        if len(self.buffer_bloques) > self.n_buffer:
            self.buffer_bloques.pop(0)

    def _buscar_disparo(self, bloque):
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

        audio_total_disponible = np.concatenate([contexto_previo, bloque]) if len(contexto_previo) else bloque
        self.evento_audio = audio_total_disponible[max(0, inicio_absoluto_muestra):]
        self.muestras_desde_ultima_actividad = 0
        self._chequear_fin_o_continuar(bloque_actual_ya_incluido=True)

    def _continuar_evento(self, bloque):
        self.evento_audio = np.concatenate([self.evento_audio, bloque])
        self._chequear_fin_o_continuar(bloque_actual_ya_incluido=True)

    def _chequear_fin_o_continuar(self, bloque_actual_ya_incluido):
        mascara, tiempos = self.detector.mascara_actividad(self.evento_audio)

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
            mascara, tiempos = self.detector.mascara_actividad(audio_final)
            if mascara.any():
                ultimo_frame_activo = len(mascara) - 1 - int(np.argmax(mascara[::-1]))
                fin_s = tiempos[ultimo_frame_activo] + (self.detector.hop / SR)
                margen_s = self.config['SILENCIO_FIN_EVENTO_S'] * (self.config['MARGEN_FINAL_PCT'] / 100.0)
                fin_muestra = min(len(audio_final), int((fin_s + margen_s) * SR))
                audio_final = audio_final[:fin_muestra]
        self.eventos_terminados.append(audio_final)
        self.evento_audio = None

    def finalizar(self):
        """Llamar al terminar el stream (o al apagar el dispositivo) para
        no perder un evento que haya quedado abierto sin llegar a cerrarse
        por silencio ni por tope."""
        if self.evento_audio is not None:
            self._cerrar_evento(recortar_silencio_final=False)

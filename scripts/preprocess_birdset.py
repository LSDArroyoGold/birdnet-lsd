"""Reimplementacion SIN torch del preprocesamiento audio->espectrograma
de BirdSet EfficientNetB1 (EfficientnetBirdsetPreprocessor), usando
librosa. Orden de operaciones confirmado leyendo
bmz_birdset/birdset_preprocessing.py (BirdsetPreprocessor.pipeline) +
bmz_birdset_efficientnetB1.py (EfficientnetBirdsetPreprocessor, que
sobreescribe los parametros de to_spec/to_mel/power_to_db/rescale):

  1. trim/pad a sample_duration exacto (aca: el llamador ya entrega el
     array de la duracion correcta, 5s @ 32kHz -- no reimplementado aca)
  2. resample a 32000 Hz (idem, ya lo hace el llamador)
  3. to_spec: espectrograma de potencia (torchaudio.transforms.Spectrogram,
     n_fft=2048, hop_length=256, power=2.0) -- defaults confirmados en el
     codigo instalado: win_length=n_fft, window=hann, center=True,
     pad_mode='reflect', normalized=False, onesided=True
  4. to_mel: banco de filtros mel (torchaudio.transforms.MelScale,
     n_mels=256, n_stft=1025, sample_rate=32000) -- defaults confirmados:
     f_min=0.0, f_max=sample_rate/2=16000, norm=None, mel_scale='htk'
     (NO 'slaney' -- default de librosa es htk=False/norm='slaney', hay
     que forzar explicitamente htk=True, norm=None para que coincida)
  5. power_to_db: clase PowerToDB custom (birdset_preprocessing.py) con
     ref=1.0, amin=1e-10, top_db=80 -- formula IDENTICA a
     librosa.power_to_db con los mismos parametros (verificado comparando
     el codigo fuente de las dos)
  6. rescale: (x - input_mean) / input_std, con input_mean=-4.268,
     input_std=4.569 (EfficientnetBirdsetPreprocessor)

Devuelve un array (1, 1, n_mels, n_frames) -- mismo shape que
'pixel_values' que espera el modelo (batch, canal, alto, ancho).
"""
import numpy as np
import librosa

N_FFT = 2048
HOP_LENGTH = 256
POWER = 2.0
N_MELS = 256
SAMPLE_RATE = 32000
F_MIN = 0.0
F_MAX = SAMPLE_RATE / 2  # 16000, default de torchaudio MelScale (f_max=None -> sample_rate/2)
TOP_DB = 80.0
REF = 1.0
AMIN = 1e-10
INPUT_MEAN = -4.268
INPUT_STD = 4.569

_mel_basis = None


def _cargar_mel_basis():
    global _mel_basis
    if _mel_basis is None:
        # htk=True, norm=None: replica exactamente los defaults de
        # torchaudio.transforms.MelScale (mel_scale='htk', norm=None) --
        # los defaults de librosa son DISTINTOS (htk=False, norm='slaney'),
        # confirmados via inspect.signature() antes de asumir nada.
        _mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX,
            htk=True, norm=None,
        )
    return _mel_basis


def audio_a_pixel_values(audio_5s_32k):
    """audio_5s_32k: array float32/float64, exactamente sample_duration*32000
    muestras (el llamador ya hizo trim/pad/resample). Devuelve
    (1, 1, n_mels, n_frames) listo para 'pixel_values' del modelo ONNX."""
    audio = np.asarray(audio_5s_32k, dtype=np.float32)

    stft = librosa.stft(
        audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT,
        window="hann", center=True, pad_mode="reflect",
    )
    power_spec = np.abs(stft) ** POWER  # power=2.0 -> magnitud al cuadrado

    mel_basis = _cargar_mel_basis()
    mel_spec = mel_basis @ power_spec  # (n_mels, n_frames)

    # PowerToDB custom == misma formula que librosa.power_to_db con estos
    # parametros (ref=1.0 escalar -> el segundo termino da 0)
    log_mel = librosa.power_to_db(mel_spec, ref=REF, amin=AMIN, top_db=TOP_DB)

    rescaled = (log_mel - INPUT_MEAN) / INPUT_STD

    return rescaled[np.newaxis, np.newaxis, :, :].astype(np.float32)

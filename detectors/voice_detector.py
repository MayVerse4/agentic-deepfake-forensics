"""AI-cloned voice detector: Gustking/wav2vec2-large-xlsr-deepfake-audio-classification (first 6 s, 16 kHz).

Why this model alone (eval_voice.py, eval_omnifake_ood.py): it was the most robust of six Hugging Face detectors.
In-the-Wild AUC 0.947; OmniFake-OOD (CVPR 2026, unseen sources) AUC 0.978. The former partner
(mo-thecreator) scored 0.950 on In-the-Wild but collapsed to 0.654 on OmniFake-OOD and pulled the pair down to 0.889.

The raw score is compressed (it never exceeds ~0.92), so it is calibrated (Platt scaling on the logit, fitted on
In-the-Wild, checked on OmniFake-OOD: mean calibrated P real 0.20/0.16, fake 0.80/0.80):
  P >= 0.82 -> "Likely AI-Cloned Voice"   (ITW 70% caught at 2.7% real flagged; OmniFake 66% at 1.2%)
  P >= 0.50 -> "Possible AI-Cloned Voice" (ITW 78% caught at 4.7%; OmniFake 80% at 2.7%)
"""
import io

import numpy as np
import soundfile as sf
import torch

SR, SECONDS = 16000, 6
MODEL = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
CAL_A, CAL_B = 1.586, 2.195          # P = sigmoid(A * logit(raw) + B)
VOICE_STRONG, VOICE_WEAK = 0.82, 0.50


class VoiceDetector:
    def __init__(self, device=None):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL)
        self.model = AutoModelForAudioClassification.from_pretrained(MODEL, use_safetensors=True).to(self.device).eval()
        names = {i: str(n).lower() for i, n in self.model.config.id2label.items()}
        fake = [i for i, n in names.items() if any(k in n for k in ("fake", "spoof", "synthetic", "deepfake"))]
        if len(fake) != 1:
            raise RuntimeError(f"cannot tell which output is 'fake': {names}")
        self.fake_index = fake[0]

    @staticmethod
    def load_audio(path_or_bytes):
        src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
        try:
            wav, sr = sf.read(src, dtype="float32")
        except Exception:                      # mp3/m4a and other containers
            import librosa
            wav, sr = librosa.load(src, sr=SR, mono=True)
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav[:SR * SECONDS]

    @torch.no_grad()
    def score_audio(self, path_or_bytes):
        wav = self.load_audio(path_or_bytes)
        if wav.size < SR // 2:
            return None                          # under half a second: nothing to judge
        x = self.fe(wav, sampling_rate=SR, return_tensors="pt").to(self.device)
        raw = float(torch.softmax(self.model(**x).logits[0].float(), -1)[self.fake_index])
        r = min(max(raw, 1e-6), 1 - 1e-6)
        p = float(1 / (1 + np.exp(-(CAL_A * np.log(r / (1 - r)) + CAL_B))))
        if p >= VOICE_STRONG:
            verdict, conf = "Likely AI-Cloned Voice", ("High" if p >= 0.95 else "Medium")
        elif p >= VOICE_WEAK:
            verdict, conf = "Possible AI-Cloned Voice", "Low"
        else:
            verdict, conf = "Likely Authentic Speech", ("Medium" if p <= 0.20 else "Low")
        return {"prob_ai": round(p, 4), "raw_score": round(raw, 4), "verdict": verdict, "confidence": conf,
                "gustking_score": round(raw, 4)}

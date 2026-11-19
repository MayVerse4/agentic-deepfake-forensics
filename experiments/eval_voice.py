"""Voice-clone (audio deepfake) detectors from Hugging Face, benchmarked on In-the-Wild
(Mueller et al., Interspeech 2022: real vs AI-cloned speech of 58 celebrities/politicians, collected online),
via SpeechAntiSpoofingBenchmarks/InTheWild shard 0 (3,973 clips) in data/benchmarks/audio/.

Every model scores the same balanced sample (first 6 s of each clip, 16 kHz). Reported: AUC, EER (equal error
rate, the standard anti-spoofing metric: lower is better), and fakes caught when 5% of real voices are flagged.
Run: .venv-win\\Scripts\\python.exe experiments\\eval_voice.py [n_per_class]
Writes results/benchmarks/voice_eval.csv, voice_summary.json, BENCHMARKS.md, and a table in results/RESULTS.md.
"""
import io, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq, soundfile as sf, torch
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(os.path.dirname(HERE), "results", "benchmarks")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 750
SR, SECONDS = 16000, 6
MODELS = ["mo-thecreator/Deepfake-audio-detection", "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
          "MelodyMachine/Deepfake-audio-detection-V2", "Hemgg/Deepfake-audio-detection",
          "Vansh180/deepfake-audio-wav2vec2", "Bisher/wav2vec2_ASV_deepfake_audio_detection"]


def load_clips():
    t = pq.read_table(os.path.join(os.path.dirname(HERE), "data", "benchmarks", "audio", "itw_shard0.parquet")).to_pandas()
    t = pd.concat([t[t.label == 0].sample(min(N, (t.label == 0).sum()), random_state=0),
                   t[t.label == 1].sample(min(N, (t.label == 1).sum()), random_state=0)])
    clips = []
    for a in t.audio:
        wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        clips.append(wav[:SR * SECONDS])
    return t.path.tolist(), t.label.values, clips          # label 1 = spoof (AI voice)


def fake_index(model):
    names = {i: str(n).lower() for i, n in model.config.id2label.items()}
    hits = [i for i, n in names.items() if any(k in n for k in ("fake", "spoof", "synthetic", "ai", "deepfake"))]
    if len(hits) == 1:
        return hits[0], names
    reals = [i for i, n in names.items() if any(k in n for k in ("real", "bona", "human", "genuine"))]
    if len(reals) == 1 and len(names) == 2:
        return 1 - reals[0], names
    return None, names


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr)); return (fpr[i] + fnr[i]) / 2


paths, y, clips = load_clips()
print(f"In-the-Wild sample: {int((y == 0).sum())} real + {int((y == 1).sum())} AI-cloned clips (first {SECONDS} s)")
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
dev = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = os.path.join(OUT, "voice_eval.csv")
scores, rows = pd.DataFrame({"path": paths, "label": y}), []
if os.path.exists(CACHE):                                # resume: reuse models already scored
    old = pd.read_csv(CACHE)
    if list(old.path) == paths:
        scores = old
for name in MODELS:
    if name in scores.columns:
        s = scores[name].values
        fpr, tpr, th = roc_curve(y, s)
        rows.append({"model": name, "auc": roc_auc_score(y, s), "eer": eer(y, s),
                     "caught_at_5pct_fpr": float(tpr[np.where(fpr <= 0.05)[0][-1]]),
                     "acc_at_half": float(((s > 0.5) == y).mean()), "labels": None})
        print(f"  {name:62s} (cached) AUC {rows[-1]['auc']:.3f}", flush=True)
        continue
    try:
        fe = AutoFeatureExtractor.from_pretrained(name)
        model = AutoModelForAudioClassification.from_pretrained(name, use_safetensors=True).to(dev).eval()
    except Exception as e:
        print(f"  {name}: could not load ({type(e).__name__})"); continue
    fi, labels = fake_index(model)
    if fi is None:
        print(f"  {name}: unclear labels {labels}, skipped"); continue
    s = []
    with torch.no_grad():
        for w in clips:
            x = fe(w, sampling_rate=SR, return_tensors="pt").to(dev)
            s.append(float(torch.softmax(model(**x).logits[0].float(), -1)[fi]))
    s = np.array(s); scores[name] = s
    scores.to_csv(CACHE, index=False)                    # save after every model
    fpr, tpr, th = roc_curve(y, s)
    caught = float(tpr[np.where(fpr <= 0.05)[0][-1]])
    rows.append({"model": name, "auc": roc_auc_score(y, s), "eer": eer(y, s), "caught_at_5pct_fpr": caught,
                 "acc_at_half": float(((s > 0.5) == y).mean()), "labels": labels})
    print(f"  {name:62s} AUC {rows[-1]['auc']:.3f}  EER {rows[-1]['eer'] * 100:5.1f}%  caught@5% {caught * 100:5.1f}%", flush=True)
    del model; torch.cuda.empty_cache()

scores.to_csv(os.path.join(OUT, "voice_eval.csv"), index=False)
res = pd.DataFrame(rows).sort_values("auc", ascending=False)
from experiments.bench_record import save
save({"benchmark": f"Voice-clone detectors on In-the-Wild (shard 0 sample: {int((y == 0).sum())} real + {int((y == 1).sum())} fake, first {SECONDS} s)",
      **{r.model: f"AUC {r.auc:.3f}, EER {r.eer * 100:.1f}%, caught {r.caught_at_5pct_fpr * 100:.1f}% at 5% real flagged, "
                  f"acc@0.5 {r.acc_at_half * 100:.1f}%" for r in res.itertuples()},
      "per_clip_csv": "voice_eval.csv"}, OUT, "voice_summary.json")
with open(os.path.join(os.path.dirname(OUT), "RESULTS.md"), "a", encoding="utf8") as f:
    f.write(f"\n## Voice-clone detection (In-the-Wild: {int((y == 0).sum())} real vs {int((y == 1).sum())} AI-cloned voices)\n\n"
            "| Model | AUC | EER (lower = better) | AI voices caught at 5% real flagged |\n|---|---|---|---|\n")
    for r in res.itertuples():
        f.write(f"| {r.model} | {r.auc:.3f} | {r.eer * 100:.1f}% | {r.caught_at_5pct_fpr * 100:.1f}% |\n")
print("\ntable added to results/RESULTS.md")

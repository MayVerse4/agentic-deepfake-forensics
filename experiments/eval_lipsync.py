"""Lip-sync deepfakes on LipSync-A (She et al., ICML 2026; open on Hugging Face). Paired real/fake clips per generator.
Four fixed detectors (no refitting on this data, so AUC needs no held-out split):
  syncnet  SyncNet (Chung & Zisserman 2016; MIT)         audio-lip sync confidence; fake score = -confidence
  lipinc   LIPINC-V2 (Datta et al. 2025)                 visual lip-inconsistency; fake score = 1 - P(real)
  lipda    LipDA audio-visual (She et al., ICML 2026; MIT) fake probability. Trained on LipSync-A's older
           generators (wav2lip, DINet, IP_LAP, sadtalker, ...), so ONLY generators outside its training
           split are a fair test ("unseen" column); the rest are reported but flagged as seen.
  face     this system's face pipeline (spatial+frequency+temporal fusion), main face track
Each detector caches its scores in results/benchmarks/lipsync/<name>.csv, so a crash never repeats work.
Run: .venv-win\\Scripts\\python.exe experiments\\eval_lipsync.py [syncnet lipinc lipda face]   (default: all, then report)"""
import os, sys, glob, json, subprocess, shutil, tempfile
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
D = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "lipsync_a"); OUT = os.path.join(REPO, "results", "benchmarks", "lipsync"); os.makedirs(OUT, exist_ok=True)
PY = sys.executable
LIPDA_TRAIN = {"wav2lip", "dinet", "ip_lap", "sadtalker", "makeittalk", "dreamtalk", "tpsm", "lia", "facevid", "x2face"}


def clips():
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "*", "[01]_*", "*.mp4"))):
        p = f.split(os.sep); g, lab = p[-3], int(p[-2][0])
        rows.append(dict(file=os.path.relpath(f, D), gen=g, label=lab, name=f"{g}_{lab}_{os.path.splitext(p[-1])[0]}"))
    return pd.DataFrame(rows)


def todo(name, df):
    path = os.path.join(OUT, name + ".csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["name", "score"])
    return path, done, df[~df.name.isin(done.name)]


def save(path, done, new):
    pd.concat([done, pd.DataFrame(new)], ignore_index=True).to_csv(path, index=False)


# ---------------- SyncNet ----------------
def run_syncnet(df):
    path, done, rest = todo("syncnet", df)
    if rest.empty: return
    import torch, python_speech_features, imageio_ffmpeg
    from scipy.io import wavfile
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "third_party", "syncnet"))
    from SyncNetModel import S
    from SyncNetInstance import calc_pdist
    FF = imageio_ffmpeg.get_ffmpeg_exe(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = S(num_layers_in_fc_layers=1024).to(dev).eval()
    net.load_state_dict(torch.load(os.path.join(os.path.dirname(HERE), "weights", "pretrained", "syncnet", "syncnet_v2.model"), map_location=dev))
    new = []
    for r in rest.itertuples():
        f = os.path.join(D, r.file)
        try:
            raw = subprocess.run([FF, "-v", "error", "-i", f, "-r", "25", "-vf", "scale=224:224", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                                 capture_output=True, check=True).stdout
            v = np.frombuffer(raw, np.uint8).reshape(-1, 224, 224, 3)
            with tempfile.TemporaryDirectory() as t:
                w = os.path.join(t, "a.wav")
                subprocess.run([FF, "-v", "error", "-y", "-i", f, "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", w], check=True)
                sr, a = wavfile.read(w)
            mf = np.stack([np.array(i) for i in zip(*python_speech_features.mfcc(a, sr))])
            n = min(len(v), len(a) // 640) - 5
            with torch.no_grad():
                im = torch.from_numpy(v.astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0); cc = torch.from_numpy(mf.astype(np.float32))[None, None]
                fi, fa = [], []
                for i in range(0, n, 64):
                    k = range(i, min(n, i + 64))
                    fi.append(net.forward_lip(torch.cat([im[:, :, j:j + 5] for j in k]).to(dev)).cpu())
                    fa.append(net.forward_aud(torch.cat([cc[:, :, :, j * 4:j * 4 + 20] for j in k]).to(dev)).cpu())
                d = torch.stack(calc_pdist(torch.cat(fi), torch.cat(fa), vshift=15), 1).mean(1)
            new.append(dict(name=r.name, score=-float(torch.median(d) - d.min())))
        except Exception as e:
            print("syncnet fail", r.file, str(e)[:100]); new.append(dict(name=r.name, score=np.nan))
        if len(new) % 50 == 0: print("syncnet", len(new), "/", len(rest), flush=True)
    save(path, done, new)


# ---------------- LIPINC-V2 ----------------
def run_lipinc(df):
    path, done, rest = todo("lipinc", df)
    if rest.empty: return
    src = os.path.join(os.path.dirname(HERE), "third_party", "lipinc"); cwd = os.getcwd(); os.chdir(src); sys.path.insert(0, src)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from utils import get_color_structure_frames
    from model import LIPINC_V2
    m = LIPINC_V2(); m.load_weights("checkpoints/Best_Weights.hdf5"); new = []
    for r in rest.itertuples():
        try:
            err, _, c, res, _, _ = get_color_structure_frames(5, os.path.join(D, r.file))
            s = np.nan if err else 1 - float(m.predict([c[None], res[None]], verbose=0)[0][1])
        except Exception as e:
            print("lipinc fail", r.file, str(e)[:100]); s = np.nan
        new.append(dict(name=r.name, score=s))
        if len(new) % 50 == 0: print("lipinc", len(new), "/", len(rest), flush=True); save(path, done, new)
    os.chdir(cwd); save(path, done, new)


# ---------------- LipDA (audio-visual) ----------------
def run_lipda(df):
    path, done, rest = todo("lipda", df)
    if rest.empty: return
    src = os.path.join(os.path.dirname(HERE), "third_party", "lipda"); cache = os.path.join(D, "lipda_cache"); os.makedirs(cache, exist_ok=True)
    lst = os.path.join(cache, "list.json")
    json.dump([dict(video=os.path.join(D, r.file), label=int(r.label), name=r.name) for r in rest.itertuples()], open(lst, "w"))
    import imageio_ffmpeg  # LipDA calls plain "ffmpeg": expose the bundled binary under that name
    tools = os.path.join(os.path.dirname(HERE), "tools"); os.makedirs(tools, exist_ok=True)
    if not os.path.exists(os.path.join(tools, "ffmpeg.exe")): shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), os.path.join(tools, "ffmpeg.exe"))
    env = dict(os.environ, PATH=tools + os.pathsep + os.environ["PATH"], PYTHONIOENCODING="utf-8")
    for s in ("preprocess/extract_mesh.py", "preprocess/extract_audio.py"):
        subprocess.run([PY, s, "--list", lst, "--cache_dir", cache, "--workers", "4"], cwd=src, env=env, check=True)
    sys.path.insert(0, src)
    import torch
    from torch.utils.data import DataLoader
    from common.clip_dataset import ClipAVDataset
    from common.utils import load_config
    from audiovisual.model import AVTSTAN
    from audiovisual.train_loop import av_collate
    cfg = load_config(os.path.join(src, "configs", "default.yaml")); dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ClipAVDataset(lst, cache, cfg, task="audiovisual", split="test", require_audio=True, max_frames=int(cfg["audiovisual"]["max_frames"]), verbose=True)
    m = AVTSTAN(num_classes=2).to(dev).eval()
    sd = torch.load(os.path.join(src, "weights", "audiovisual_deepfake_detector.pth"), map_location=dev)["model"]
    m.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})
    new, seen = [], set()
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32, shuffle=False, collate_fn=av_collate):
            p = torch.softmax(m(b["landmarks"].to(dev), b["lip_roi"].to(dev), b["mfcc"].to(dev))[0], 1)[:, 1].cpu().numpy()
            for n, s in zip(b["name"], p):
                if n not in seen: seen.add(n); new.append(dict(name=n, score=float(s)))
    missing = set(rest.name) - seen
    new += [dict(name=n, score=np.nan) for n in missing]
    print("lipda scored", len(seen), "missing", len(missing)); save(path, done, new)


# ---------------- this system's face pipeline ----------------
def run_face(df):
    path, done, rest = todo("face", df)
    if rest.empty: return
    sys.path.insert(0, REPO); cwd = os.getcwd(); os.chdir(REPO)
    from agent.inference_engine import InferenceEngine
    eng = InferenceEngine(); new = []
    for r in rest.itertuples():
        try:
            t = eng.analyze(os.path.join(D, r.file))["per_track_summary"]
            s = max(t, key=lambda x: x["frames_seen"])["combined_prob_fake"] if t else np.nan
        except Exception as e:
            print("face fail", r.file, str(e)[:100]); s = np.nan
        new.append(dict(name=r.name, score=s))
        if len(new) % 50 == 0: print("face", len(new), "/", len(rest), flush=True); save(path, done, new)
    os.chdir(cwd); save(path, done, new)


def report(df):
    res = {}
    for det in ("syncnet", "lipinc", "lipda", "face"):
        p = os.path.join(OUT, det + ".csv")
        if not os.path.exists(p): continue
        m = df.merge(pd.read_csv(p), on="name").dropna(subset=["score"])
        r = {}
        groups = [("ALL", m)] + list(m.groupby("gen"))
        if det == "lipda":
            unseen = m[~m.gen.str.lower().isin(LIPDA_TRAIN)]
            groups.insert(1, ("UNSEEN_BY_LIPDA", unseen))
        for g, s in groups:
            if s.label.nunique() == 2:
                r[g] = dict(auc=round(roc_auc_score(s.label, s.score), 4), n=len(s), missing=int(df[df.gen == g].shape[0] - len(s)) if g in set(df.gen) else None,
                            seen_in_training=(g.lower() in LIPDA_TRAIN) if det == "lipda" and g in set(df.gen) else None)
        res[det] = r
    json.dump(res, open(os.path.join(OUT, "lipsync_auc.json"), "w"), indent=1)
    gens = sorted(df.gen.unique())
    print("\nAUC".ljust(18) + "".join(d.rjust(10) for d in res))
    for g in ["ALL", "UNSEEN_BY_LIPDA"] + gens:
        print(g.ljust(18) + "".join((f"{res[d][g]['auc']:.3f}" if g in res[d] else "-").rjust(10) for d in res))


if __name__ == "__main__":
    df = clips(); print(len(df), "clips:", df.groupby(["gen", "label"]).size().to_dict())
    for st in (sys.argv[1:] or ["syncnet", "lipinc", "lipda", "face"]):
        print("==", st, flush=True); globals()["run_" + st](df)
    report(df)

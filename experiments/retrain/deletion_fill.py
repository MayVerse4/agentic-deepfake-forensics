"""Deletion faithfulness test with neutral fill (ImageNet-mean grey) instead of blur, which the detector reads as a forgery cue.
Removes top-k% vs bottom-k% Grad-CAM pixels (k=10,20,30) on fakes; faithful maps -> larger P(fake) drop for top-k."""
import os, sys, json, random, numpy as np, torch
from PIL import Image
import paper_stats as P
from gradcam_faces import cam
if __name__ == "__main__":
    from spatial_model import DeepfakeSpatialModel
    res = {}
    for name, ck in (a.split("=", 1) for a in sys.argv[1:]):
        m = DeepfakeSpatialModel(); m.load_state_dict(torch.load(os.path.join(P.REPO, ck), map_location="cpu")); m = m.cuda().eval()
        sets = P.data(); rnd = random.Random(0); out = {}
        for k in ("ff_val", "df40_test"):
            items = rnd.sample([i for i in sets[k] if i[1] == 1], 100); d = {f"{t}{q}": [] for t in ("top", "bottom") for q in (10, 20, 30)}
            for path, *_ in items:
                im = Image.open(path).convert("RGB").resize((224, 224)); x = P.TF(im).unsqueeze(0).cuda()
                p, c = cam(m, x); c = c[0].cpu().numpy(); a = np.asarray(im)
                for q in (10, 20, 30):
                    for t, mask in (("top", c >= np.quantile(c, 1 - q / 100)), ("bottom", c <= np.quantile(c, q / 100))):
                        mix = np.where(mask[..., None], np.array([124, 116, 104], np.uint8), a).astype(np.uint8)
                        with torch.no_grad(): r = torch.sigmoid(m(P.TF(Image.fromarray(mix)).unsqueeze(0).cuda())).item()
                        d[f"{t}{q}"].append(p.item() - r)
            out[k] = {t: round(float(np.mean(v)), 4) for t, v in d.items()}; print(name, k, out[k], flush=True)
        res[name] = out
    json.dump(res, open(os.path.join(P.OUT, "gradcam", "deletion_fill.json"), "w"), indent=1)

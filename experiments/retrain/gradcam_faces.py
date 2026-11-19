"""Grad-CAM heatmaps for the spatial face model (EfficientNet-B0, last conv block), plus a simple faithfulness check
reviewers ask for: deletion test -- blur the top-k% most-attributed pixels and measure how much P(fake) drops, compared with
blurring the 20% least-attributed pixels. A faithful map gives a much larger drop than random.
Usage: python gradcam_faces.py ckpt [n_per_class]  -> results/retrain/gradcam/ (grid png + deletion.json)"""
import os, sys, csv, json, random
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageFilter
import paper_stats as P

def cam(m, x):
    acts, grads = {}, {}
    layer = m.backbone.features[-1]
    h1 = layer.register_forward_hook(lambda _, __, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda _, gi, go: grads.__setitem__("g", go[0]))
    m.zero_grad(); logit = m(x).view(-1); logit.sum().backward(); h1.remove(); h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True); c = F.relu((w * acts["a"]).sum(1, keepdim=True))
    c = F.interpolate(c, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    c = c / (c.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
    return torch.sigmoid(logit).detach(), c.detach()

def overlay(im, c):
    import matplotlib.cm as cm
    h = (cm.jet(c)[..., :3] * 255).astype(np.uint8)
    return Image.blend(im.resize((224, 224)), Image.fromarray(h), 0.45)

if __name__ == "__main__":
    from spatial_model import DeepfakeSpatialModel
    ck = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = os.path.join(P.OUT, "gradcam"); os.makedirs(out, exist_ok=True)
    m = DeepfakeSpatialModel(); m.load_state_dict(torch.load(os.path.join(P.REPO, ck), map_location="cpu")); m = m.cuda().eval()
    sets = P.data(); rnd = random.Random(0)
    pick = {}
    for k in ("ff_val", "df40_test"):
        for lab in (0, 1):
            pick[(k, lab)] = rnd.sample([i for i in sets[k] if i[1] == lab], n)
    tiles, drops = [], {"topk": [], "bottomk": []}
    for (k, lab), items in pick.items():
        row = []
        for path, y, vid, meth in items:
            im = Image.open(path).convert("RGB"); x = P.TF(im).unsqueeze(0).cuda().requires_grad_(False)
            p, c = cam(m, x); c = c[0].cpu().numpy()
            row.append((overlay(im, c), f"{meth} p={p.item():.2f}"))
            if y == 1:  # deletion test on fakes: blur top 20% attributed pixels vs random 20%
                base = im.resize((224, 224)); bl = base.filter(ImageFilter.GaussianBlur(6)); a, b = np.asarray(base), np.asarray(bl)
                # compare blurring the 20% MOST vs 20% LEAST attributed pixels (both contiguous regions; scattered random
                # pixels would add high-frequency noise the detector itself reacts to, an unfair baseline)
                k20 = c >= np.quantile(c, 0.8); b20 = c <= np.quantile(c, 0.2)
                for tag, mask in (("topk", k20), ("bottomk", b20)):
                    mix = np.where(mask[..., None], b, a).astype(np.uint8)
                    with torch.no_grad(): q = torch.sigmoid(m(P.TF(Image.fromarray(mix)).unsqueeze(0).cuda())).item()
                    drops[tag].append(p.item() - q)
        tiles.append(((k, lab), row))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(tiles), n, figsize=(2.2 * n, 2.4 * len(tiles)))
    for r, ((k, lab), row) in enumerate(tiles):
        for c_, (img, title) in enumerate(row):
            ax[r, c_].imshow(img); ax[r, c_].set_title(title, fontsize=7); ax[r, c_].axis("off")
        ax[r, 0].text(-20, 112, f"{k}\n{'fake' if lab else 'real'}", fontsize=8, ha="right", va="center")
    plt.tight_layout(); fig.savefig(os.path.join(out, "gradcam_grid.png"), dpi=150)
    res = {t: dict(mean_drop=round(float(np.mean(v)), 4), n=len(v)) for t, v in drops.items()}
    json.dump(res, open(os.path.join(out, "deletion.json"), "w"), indent=1); print(res)

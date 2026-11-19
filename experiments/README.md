# Experiments and evidence-tool notes

Everything added after the original pipeline lives here, so it stays separate from the older code.

## Why these were added
The three trained detectors (spatial, frequency, temporal) only ever look at the **cropped face**, and were
trained on FaceForensics++ face swaps. So the original app could not:
- see an edit **outside the face** (background, objects, text);
- judge an image **with no face** (e.g. a fake city photo). It said "Cannot Determine" unless the
  file's metadata declared AI generation, and stripping the metadata defeated even that.

Two pixel-only tools close those gaps. Neither needs metadata.

| Tool | Answers | Model | Licence |
|---|---|---|---|
| `localization.py` | **Which part** of the image was edited? | TruFor (Guillaro et al., CVPR 2023) | non-profit / research only |
| `ai_image_detector.py` | Was the **whole** picture made by an AI generator? | Community Forensics (Park & Owens, CVPR 2025), ViT-S/384 | MIT |

Together with the face models, each tool covers a different kind of fake:
**face swap in real footage** -> face models; **local edit / AI inpainting** -> TruFor;
**fully generated picture** -> Community Forensics.

## How the agent decides (`backend_hooks.py`)
Both tools run on every upload (images; videos via key frames: 4 for TruFor, 8 for the AI detector).

**No face found**, checked in this order (a metadata AI declaration still comes first):
1. TruFor >= 0.80 and reliability >= 0.70 -> **Edited region detected** (High if >= 0.95, else Medium)
2. AI detector >= 0.85 -> **Likely AI-generated** (Medium; never High, because detectors miss new generators)
3. TruFor 0.50-0.80 (images only) -> **Possible edit** (Low)
4. AI detector 0.50-0.85 -> **Possibly AI-generated** (Low)
5. otherwise -> **No traces found** (Low; explicitly "not proof of authenticity")

**Face found:** the trained face models decide, as validated in the paper. The agent adds a cross-check
warning if TruFor flags a region (>= 0.50 for images, >= 0.80 for videos) or the AI detector >= 0.50.

**Videos use a stricter TruFor threshold (0.80):** TruFor is trained on photos. On compressed video
frames it scored a real broadcast video 0.54 (on its genuinely composited TV-graphics overlay) and a
face-swap video only 0.48, so weak video scores are not treated as evidence.

## Verified results
| Check | Result |
|---|---|
| Face models end-to-end (16 FF++ videos re-run from raw video) | scores match the paper's saved values (max diff 0.003); 16/16 correct |
| Live app: city photo with pasted patch, no face/metadata | Edited region detected; box within 2-3 px of the true edit |
| Live app: untouched city photo | No traces found (Low) |
| Live app: DALL-E 2 image, metadata stripped | Likely AI-generated (P = 1.00) |
| Live app: real / face-swap FF++ videos | Likely Real / Likely Fake |
| TruFor on CocoGlide (512 real + 512 AI-inpainted) | see `../output/benchmarks/` (eval_localization.py) |
| AI detector on Community Forensics eval sample | see `../output/benchmarks/` (eval_ai_detector.py) |

## Update: three-tool edit detection, EXIF consistency, AI watermarks
**Images** now run TruFor + Mesorch (Zhu et al., AAAI 2025, MIT) + the AI detector, combined by a
logistic regression fitted on CocoGlide (`COMBO_*` in `backend_hooks.py`). On held-out images it catches
47% of AI-inpainted edits at 3% real photos flagged (TruFor alone: 29% at 5%). Mesorch draws the outline.
Videos are unchanged (the combiner was only validated on images).

**B-Free** (`bfree_detector.py`; Guillaro et al., CVPR 2025; non-profit licence) is added for **images**:
whole-image AI = Community Forensics + B-Free (89.8% caught at 4.7% real flagged vs 87.6% at 6.1%), and as a
4th tool in the edit score (CocoGlide: 66% of AI edits at 4% vs 51% at 6%). Not used on video (no gain:
86.8% vs 85.8%). A high whole-image AI score without a strong TruFor edit is labelled AI-generated, not edited.
All benchmark files are in `../output/benchmarks/`.

**EXIF consistency** (`exif_consistency.py`): thumbnail vs picture, recorded vs actual size, dates,
standard-libjpeg tables on a camera JPEG, missing MakerNote. A thumbnail/size contradiction gives the
verdict "Edited (metadata evidence)" when the pixel tools found nothing.

**Invisible AI watermarks** (`watermark_check.py`, `invisible-watermark`): Stable Diffusion v1/v2 and SDXL.
0/518 clean images flagged; a hit counts like a metadata AI declaration. JPEG re-saving removes the mark.

**Tested and not used** (results kept in `../output/benchmarks/`): eye-reflection physics check (AUC 0.46 at
1024 px), NSG-VD physics video detector (AUC 0.59 once all clips share one format; its 0.79 before was a
resolution shortcut). SAFIRE weights are not downloadable.

**Reverse image search** (`backend/source_verification.py`, Google Lens via SerpAPI) is built but off:
put `SERPAPI_KEY=...` (free plan, 250 searches/month) in the project-root `.env` to enable it.

## Files
| File | Purpose |
|---|---|
| `localization.py` | `Localizer().analyze(img)` -> score, regions, overlay, heatmap, reliability |
| `ai_image_detector.py` | `AIImageDetector().score_image(img)` / `.score_video(path)` -> P(AI-generated) |
| `backend_hooks.py` | glue for `backend/forensics_api.py`; all thresholds and decision rules |
| `eval_localization.py` | TruFor benchmark on CocoGlide |
| `eval_ai_detector.py` | AI-detector benchmark on a random sample of CommunityForensics-Eval |
| `api_smoke_test.py` | uploads files through the running app's API; prints verdict, steps, explanation |
| `trufor_src/`, `commfor_src/` | official code of both models (unchanged clones) |
| `weights/` | TruFor weights (268 MB); Community Forensics downloads itself from Hugging Face on first use |
| `data/` | CocoGlide benchmark images (downloaded) |
| `../output/benchmarks/` | benchmark CSVs |

Both tools import top-level modules named `config`, `models`, `utils`, the same names the project uses.
`localization.py` and `ai_image_detector.py` swap them in only while they run, so neither side breaks.

### Changes outside this folder (kept small)
- `backend/forensics_api.py`: `_pixel_tools()` step; `decide_no_face()` uses the new rules; cross-check notes
- `backend/llm.py`: explanation prompts for the new verdicts
- `frontend/app/sections/ForensicsSection.tsx`: AI-check and localization panels; colours for the new verdicts
- `requirements-windows.txt`: `timm`, `yacs`, `huggingface_hub`, `transformers`, `torchmetrics`, `datasets`
  (`protobuf` must stay < 5 for TensorFlow; `wandb` is deliberately not installed)

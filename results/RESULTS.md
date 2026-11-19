# Results: all benchmarks in one place

Plain tables of every test run on this project. The raw numbers, per-file CSVs and run logs are in
`benchmarks/` (and `benchmarks/BENCHMARKS.md`, the append-only log). "Caught" = share of fakes detected;
"real flagged" = share of genuine media wrongly flagged (false alarms). Unless noted, thresholds were chosen
on one half of the data and measured on the other half (5 random splits).

## What the app uses now

| Media | Check | Tools |
|---|---|---|
| Face in image/video | Face swap | Spatial + Frequency + Temporal face models -> Fusion (original project) |
| Image | Edited region | TruFor + Mesorch + Community Forensics + B-Free (combined); Mesorch draws the box; on small images capped at 'Possible edit' unless TruFor and Mesorch are both >= 0.95 |
| Image | Fully AI-generated | Community Forensics + B-Free + GAPL (combined); small images (<512 px): Community Forensics + GAPL |
| Video | Fully AI-generated | Community Forensics (8 frames) |
| Any file | Provenance | C2PA, IPTC, EXIF device/software, AI-tool signatures |
| JPEG photo | EXIF consistency | thumbnail vs picture, size, dates, compression tables |
| Image | Invisible AI watermark | Stable Diffusion v1/v2 + SDXL decoder |
| Any file | Reverse image search | Google Lens via SerpAPI |
| Audio / Voice clip | AI voice clone | Gustking wav2vec2 XLS-R alone, calibrated (mo-thecreator dropped: fails on unseen voices) |

## Face models (original project, re-verified)

| Test | Result |
|---|---|
| 16 FF++ videos re-run from raw video | scores match the paper's saved values (max difference 0.003); 16/16 correct |

## AI-edited images (CocoGlide: 512 real + 512 AI-inpainted)

| Tools | AUC | Edits caught | Real flagged |
|---|---|---|---|
| TruFor alone (first version of the app) | 0.737 | 29.0% | 5.0% |
| Mesorch alone | 0.740 | 23.9% | 4.5% |
| Community Forensics alone | 0.755 | 36.8% | 4.9% |
| TruFor + Community Forensics | 0.798 | 44.5% | 4.6% |
| TruFor + Mesorch + Community Forensics | 0.822 | 50.6% | 5.9% |
| B-Free alone | 0.893 | 64.4% | 4.4% |
| **All four (app now)** | **0.886** | **66.3%** | **4.2%** |

Region outline quality (pixel F1, higher = better): TruFor 0.36, **Mesorch 0.45** (used for the box).

## Fully AI-generated images (CommunityForensics-Eval sample: 899 images, 17 generators)

| Tools | AUC | AI images caught | Real flagged |
|---|---|---|---|
| Community Forensics alone | 0.966 | 87.6% | 6.1% |
| B-Free alone | 0.924 | 71.6% | 4.7% |
| **Both combined (app now)** | **0.967** | **89.8%** | **4.7%** |

Note: this set comes from Community Forensics' own evaluation data (home advantage for it).

## AI-generated videos (GenVideo: 519 real vs 585 AI, 10 generators, all re-encoded to one format)

| Tools | AUC | AI videos caught | Real flagged |
|---|---|---|---|
| **Community Forensics (app now)** | **0.979** | **85.8%** | **4.0%** |
| B-Free alone | 0.894 | 65.4% | 4.2% |
| Both combined | 0.947 | 86.8% | 4.3% (no real gain, not used) |
| NSG-VD physics detector | 0.589 | 7.3% | 4.1% (not used) |

Before re-encoding, NSG-VD looked good (AUC 0.79): it was recognising each generator's video size, not the fake.

## Metadata checks

| Check | Result |
|---|---|
| EXIF consistency (4 devices: originals, re-saves, crops, local edits with EXIF kept) | 16/16 correct |
| SD invisible watermark | 0/518 clean images flagged; untouched watermarks found 66-78%; any JPEG re-save removes it |

## Tested and rejected

| Idea | Result |
|---|---|
| Eye-reflection physics (1024 px faces) | AUC 0.46 (no better than chance) |
| NSG-VD video physics | AUC 0.59 after removing the format shortcut |
| Gemini SynthID (reverse-engineered) | flagged 38 of 150 real photos; not usable |
| SAFIRE | weights not downloadable |

## End-to-end app test (latest)

| File | Verdict | Correct? |
|---|---|---|
| City photo with pasted patch | Edited region detected (box on the edit) | yes |
| DALL-E image, metadata stripped | Likely AI-generated | yes |
| Canon photo edited, original EXIF kept | Edited (metadata evidence) | yes |
| FF++ real video / face-swap video | Likely Real / Likely Fake | yes / yes |
| Gemini logo image | Likely AI-generated | yes |
| Gemini photo of a presenter | Inconclusive | no (newest generator not recognised) |

## Voice-clone detection (In-the-Wild: 750 real vs 750 AI-cloned voices)

| Model | AUC | EER (lower = better) | AI voices caught at 5% real flagged |
|---|---|---|---|
| **Combined ensemble (mo-thecreator + Gustking)** | **0.969** | **9.5%** | **87.1%** |
| mo-thecreator/Deepfake-audio-detection | 0.947 | 11.4% | 80.5% |
| Gustking/wav2vec2-large-xlsr-deepfake-audio-classification | 0.942 | 13.1% | 80.5% |
| Bisher/wav2vec2_ASV_deepfake_audio_detection | 0.843 | 22.1% | 44.5% |
| Hemgg/Deepfake-audio-detection | 0.793 | 26.7% | 21.9% |
| Vansh180/deepfake-audio-wav2vec2 | 0.595 | 45.6% | 5.3% |
| MelodyMachine/Deepfake-audio-detection-V2 | 0.417 | 53.3% | 0.0% |

## Independent benchmark: AIGIBench (1,000 images: 500 real vs 500 AI, 10 subsets)

| Tools | AUC | AI images caught | Real flagged |
|---|---|---|---|
| Community Forensics alone | 0.965 | 79.8% | 3.7% |
| B-Free alone | 0.958 | 81.3% | 5.1% |
| **Both combined** | **0.981** | **88.0%** | **3.9%** |

### Per-generator / subset breakdown (share caught @ 5% real flagged)

| Generator / Subset | Community Forensics | B-Free | Both Combined |
|---|---|---|---|
| CommunityAI | 64.1% | 80.0% | **82.4%** |
| DALLE-3 | 99.2% | 100.0% | **100.0%** |
| FLUX1-dev | 76.6% | 36.1% | **61.6%** |
| Imagen3 | 47.5% | 68.9% | **65.6%** |
| InstantID | 75.5% | 98.5% | **100.0%** |
| Midjourney | 82.1% | 85.5% | **87.8%** |
| SD3 | 89.9% | 100.0% | **100.0%** |
| SDXL | 100.0% | 100.0% | **100.0%** |
| SocialRF | 64.7% | 81.2% | **84.4%** |
| StyleGAN-XL | 100.0% | 66.5% | **100.0%** |

## Voice: combinations (held-out half, 5 splits, threshold for <=5% real flagged)

| Detectors | AUC | EER | AI voices caught | Real flagged |
|---|---|---|---|---|
| mo-thecreator alone | 0.950 | 11.4% | 79.7% | 4.8% |
| **mo-thecreator + Gustking (app)** | **0.974** | **8.7%** | **89.1%** | **4.6%** |
| 4 models (mo + Gustking + Bisher + Hemgg) | 0.966 | 7.5% | 88.9% | 4.6% (no gain over 2 models) |

## Correction note (2026-11-03)
An earlier docs/scoreboard.pdf (made by another assistant) listed voice "EER 5.8%, 94.1% caught", video
"Community Forensics alone 86.8%" and metadata "100% detection". These did not match the saved test files.
The scoreboard is now generated by `new/make_scoreboard.py` directly from `output/benchmarks/`
(old copy kept in `archive/scoreboard_by_other_ai.pdf`). SPAI and GAPL were wired into the app with
untested weights and have been taken out of the decisions until benchmarked.

## Newer AI-image detectors on AIGIBench (998 images: 498 real / 500 AI)

Held-out half, 5 splits, threshold for <=5% real flagged. SPAI run with its official tool (--resize-to 1024); GAPL with the authors' checkpoint (all weights loaded).

| Detectors | AUC | AI caught | Real flagged | Accuracy | F1 |
|---|---|---|---|---|---|
| Community Forensics | 0.965 | 79.8% | 3.7% | 88.1% | 86.9% |
| B-Free | 0.958 | 81.3% | 5.1% | 88.1% | 87.2% |
| SPAI (CVPR 2025) | 0.759 | 33.2% | 4.2% | 64.4% | 48.2% |
| GAPL (CVPR 2026) | 0.968 | 84.1% | 4.3% | 89.9% | 89.3% |
| CF + B-Free (app now) | 0.981 | 88.0% | 3.9% | 92.1% | 91.8% |
| CF + B-Free + SPAI | 0.975 | 86.6% | 3.8% | 91.4% | 91.0% |
| CF + B-Free + GAPL | 0.990 | 95.2% | 4.4% | 95.4% | 95.4% |
| All four | 0.989 | 94.3% | 4.2% | 95.0% | 95.0% |

Per subset, AI images caught at score > 0.5 (CF / B-Free / SPAI / GAPL):

| Subset | CF | B-Free | SPAI | GAPL |
|---|---|---|---|---|
| CommunityAI | 58% | 76% | 76% | 70% |
| DALLE-3 | 96% | 98% | 86% | 88% |
| FLUX1-dev | 64% | 34% | 66% | 92% |
| Imagen3 | 28% | 56% | 50% | 90% |
| InstantID | 64% | 98% | 96% | 98% |
| Midjourney | 62% | 82% | 90% | 94% |
| SD3 | 78% | 100% | 98% | 100% |
| SDXL | 100% | 100% | 100% | 100% |
| SocialRF | 58% | 80% | 62% | 88% |
| StyleGAN-XL | 100% | 48% | 100% | 100% |

GAPL on AI-edited images (CocoGlide, same protocol): GAPL alone AUC 0.866, 58.3% caught; 4 tools (app) 66.3%;
4 tools + GAPL 67.4% (+1.1 pts, within noise). GAPL helps whole-image AI detection, not local-edit detection.
SPAI does not help on either (AIGIBench AUC 0.759 alone; adding it lowers the combination).

## OmniFake-OOD (Omni-Fake, CVPR 2026; out-of-distribution test sample)
"AUC (app weights)" = the app's combination exactly as deployed (weights fitted on other datasets); the other columns refit on half of this data (5 splits, <=5% real flagged).

### Voice: REAL vs fake speech
339 real / 411 fake

| Detectors | AUC (app weights, no refit) | AUC | Caught | Real flagged | Accuracy |
|---|---|---|---|---|---|
| mo-thecreator | 0.644 | 0.654 | 0.0% | 0.0% | 44.4% |
| Gustking | 0.978 | 0.977 | 90.5% | 5.3% | 92.4% |
| mo + Gustking (app) | 0.889 | 0.971 | 88.7% | 5.2% | 91.5% |

## OmniFake-OOD (Omni-Fake, CVPR 2026; out-of-distribution test sample)
"AUC (app weights)" = the app's combination exactly as deployed (weights fitted on other datasets); the other columns refit on half of this data (5 splits, <=5% real flagged).

### Video: REAL vs fake video
106 real / 194 fake

| Detectors | AUC (app weights, no refit) | AUC | Caught | Real flagged | Accuracy |
|---|---|---|---|---|---|
| Community Forensics, 8 frames (app) | 0.766 | 0.769 | 49.4% | 5.4% | 64.5% |

## OmniFake-OOD (Omni-Fake, CVPR 2026; out-of-distribution test sample)
"AUC (app weights)" = the app's combination exactly as deployed (weights fitted on other datasets); the other columns refit on half of this data (5 splits, <=5% real flagged).

### AI image: REAL vs FULL_SYNTHETIC
268 real / 236 fake

| Detectors | AUC (app weights, no refit) | AUC | Caught | Real flagged | Accuracy |
|---|---|---|---|---|---|
| Community Forensics | 0.870 | 0.878 | 52.8% | 5.2% | 74.5% |
| B-Free | 0.526 | 0.544 | 7.0% | 5.0% | 52.3% |
| GAPL | 0.850 | 0.856 | 51.3% | 4.2% | 74.2% |
| CF + B-Free (app now) | 0.756 | 0.759 | 51.3% | 4.5% | 74.1% |
| CF + B-Free + GAPL (proposed) | 0.844 | 0.860 | 60.9% | 3.5% | 79.2% |

### Edited image: REAL vs TAMPERED
268 real / 246 fake

| Detectors | AUC (app weights, no refit) | AUC | Caught | Real flagged | Accuracy |
|---|---|---|---|---|---|
| TruFor | 0.526 | 0.537 | 7.8% | 6.6% | 52.5% |
| Mesorch | 0.507 | 0.497 | 7.2% | 3.6% | 53.8% |
| 4 tools (app now) | 0.622 | 0.634 | 12.3% | 5.1% | 55.6% |
| 4 tools + GAPL |  | 0.722 | 25.5% | 7.0% | 60.9% |

## App update (2026-11-16)
| Change | Evidence |
|---|---|
| GAPL added to the AI-image check | AIGIBench caught 88.0% -> 95.2% (AUC 0.981 -> 0.990); OmniFake-OOD AUC 0.756 -> 0.844 (0.885 with the small-image branch) |
| Small images: B-Free left out of the AI-image check | B-Free AUC 0.53 on 384 px OmniFake images (needs 504 px crops) |
| Small images: edit findings capped at "Possible edit" (unless TruFor and Mesorch both >= 0.95) | TruFor/Mesorch AUC 0.53/0.51 on 384 px JPEG; no real image reached >= 0.95 on both |
| Voice: Gustking alone, calibrated | OmniFake-OOD AUC 0.978 vs 0.889 for the old pair; In-the-Wild 0.947 vs 0.974 |
| Audio uploads enabled (backend allow-list + file picker) | voice detection was unreachable from the app before; real/AI test clips now give P = 0.13 / 0.99 |
| GAPL verified in the app environment | scores identical to the official run (max difference 0.0000 on 60 images) |

## Face models vs GenD (WACV 2026), 2026-11-17

Data: 3,000 random crops from `Dataset/Test` (1,500 real / 1,500 fake; a face was found in 2,910). This is the team's
earlier non-FF++ face dataset (in-the-wild photos), **not** the FF++ data the deployed models were fine-tuned on, so
this is a cross-dataset test. Protocol as elsewhere (5 half-splits, threshold for <=5% real flagged).
Scores: `output/benchmarks/gend_frames_scores.csv` (`new/eval_gend.py frames`).

| Detector | AUC | Caught at 5% false alarms | Accuracy |
|---|---|---|---|
| Spatial (ours) | 0.416 | 4.9% | 50.1% |
| Frequency (ours) | 0.452 | 7.6% | 51.6% |
| Spatial + Frequency | 0.581 | 9.7% | 52.2% |
| **GenD** | **0.690** | **28.9%** | **61.6%** |
| Spatial + Frequency + GenD | 0.716 | 26.0% | 60.2% |

Our FF++-trained face models do no better than chance on unfamiliar faces; GenD is clearly better but far from solved.

### OmniFake-OOD videos (face pipeline + GenD), real vs tampered, videos where a face was found

Only 125 of 300 sampled videos contain a face; 79 are real/tampered (64 real, 15 tampered) -- too few for firm
conclusions. Scores: `output/benchmarks/gend_omnifake_scores.csv`.

| Detector | n | AUC |
|---|---|---|
| Spatial | 79 | 0.702 |
| Frequency | 79 | 0.578 |
| Temporal | 75 | 0.715 |
| Fusion | 79 | 0.689 |
| GenD | 75 | 0.548 |

### OmniFake-OOD talking-head videos (audiovisual config): real vs fully AI-generated

300 sampled clips (138 real / 162 AI-generated, 512x512); a face was found in 295. Frames capped at 64 per video for
memory (app default 500). Protocol: 5 half-splits, threshold for <=5% real flagged. Scores:
`output/benchmarks/gend_omnifake_av_scores.csv`.

| Detector | n | AUC | Caught at 5% false alarms | Accuracy at 0.5 |
|---|---|---|---|---|
| Spatial | 295 | 0.681 | 18.3% | 62.7% |
| Frequency | 295 | 0.460 | 6.6% | 48.5% |
| Temporal | 292 | 0.481 | 9.3% | 45.5% |
| Fusion (deployed) | 295 | 0.617 | 9.1% | 55.9% |
| GenD | 293 | 0.675 | 26.7% | 61.1% |
| Spatial+Frequency+Temporal (refit) | 292 | 0.703 | 23.4% | -- |
| **Spatial+Frequency+Temporal+GenD (refit)** | 292 | **0.748** | **38.6%** (8.2% flagged) | -- |

Fully AI-generated talking heads are not face swaps, so the FF++ models were never trained for them. Adding GenD lifts
AUC 0.703 -> 0.748, but all face detectors remain weak here; retraining on newer data is needed.

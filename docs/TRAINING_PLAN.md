# Training plan: what we can retrain, on which data, and where we can improve (2026-11-17)

Nothing here is trained yet. Access status is from a web check on 2026-11-17; licences must be read on each
official page before download. Hardware: RTX 4050 (6 GB), 16 GB RAM -> full fine-tuning of large (ViT-L) models is not
possible; parameter-efficient fine-tuning (LayerNorm-only like GenD, or LoRA like GAPL) is.

## 1. Where we stand (measured)

| Area | Best now | Weak spot (measured) |
|---|---|---|
| Face swap (FF++) | Fusion AUC 0.967 | Works only on FF++-style swaps |
| Unfamiliar faces (older dataset) | Ours+GenD AUC 0.716 | Spatial 0.416, Frequency 0.452 = chance |
| AI talking heads (OmniFake) | Ours+GenD AUC 0.748 | Frequency 0.460, Temporal 0.481, Fusion 0.617 |
| AI images (AIGIBench) | CF+B-Free+GAPL AUC 0.990 | Small images (<512 px) |
| Edited regions (CocoGlide) | 66.3% caught | TruFor alone 29.0% |
| AI video (GenVideo) | 85.8% caught | No physics/temporal check survived |
| Voice clone (In-the-Wild) | 78.4% caught, AUC 0.947 | Single model; mo-thecreator collapsed on OmniFake |
| Lip-sync fakes | not covered | -- |
| Explainability | text, time ranges, edit heatmaps | No face heatmap, no voice-segment highlight |

## 2. What we can retrain (priority order)

| # | Model | How | Train data | Fits 6 GB? | Expected gain |
|---|---|---|---|---|---|
| 1 | **Spatial (EfficientNet-B0)** | full fine-tune | FF++ + DF40 + Celeb-DF(++) + own Dataset/Train | yes | biggest: from chance to useful on new face fakes |
| 2 | **GenD (CLIP-L)** | LayerNorm-only, as in the paper | same face data | yes (fp16, batch 16) | lifts the 0.69 / 0.68 AUCs |
| 3 | Frequency (ResNet18) | full fine-tune | same face data | yes | currently below chance on new data |
| 4 | Temporal (ResNet18+LSTM) | fine-tune | video face sets (DF40 video, Celeb-DF, OmniFake-Set) | yes, short clips | currently below chance |
| 5 | **Fusion** | refit on all branch scores incl. GenD | held-out mix of all face sets | trivial | the step that gave +0.045..0.135 AUC |
| 6 | Voice (wav2vec2 XLS-R) | fine-tune top layers / LoRA | MLAAD + ASVspoof5 + SpoofCeleb | yes (short clips) | robustness to new TTS tools |
| 7 | GAPL / B-Free (AI images) | LoRA top-up | Community Forensics train, OpenFake | tight | small; already 0.990 AUC |
| 8 | Mesorch (edit localisation) | fine-tune | CocoGlide-style inpainting sets | tight | edit catch rate 66% -> higher |
| 9 | Final combiners (image, video, voice) | refit logistic stacking | all held-out scores | trivial | free gain after each retrain |

"Ours" story for the paper: items 1, 3, 4, 5 are our own architectures retrained by us; items 2, 6, 7, 8 are
fine-tunes of public models; item 9 is our own decision layer.

## 3. Datasets (latest, good)

| Dataset | Year | What | Size | Access |
|---|---|---|---|---|
| FaceForensics++ | 2019 | 4 face-manipulation methods | 1,000 real + 4,000 fake videos | form (we already have it) |
| [DF40](https://github.com/YZY-stack/DF40) | NeurIPS 2024 | 40 deepfake methods (swap, reenact, entire-face synthesis, editing) | ~10x FF++ | GitHub, form/Drive |
| [Celeb-DF++](https://arxiv.org/abs/2507.18015v1) | 2025 | 22 recent methods: swap, reenactment, talking face | large | request form |
| [MFFI](https://arxiv.org/html/2509.05592v1) | ACM MM 2025 | 50 forgery methods incl. 2025 commercial tools | large | check page |
| [DiffFace-Edit](https://arxiv.org/abs/2601.13551) | 2026 | diffusion regional face edits | -- | check page |
| Omni-Fake-Set ([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Omni-Fake_Benchmarking_Unified_Multimodal_Social_Media_Deepfake_Detection_CVPR_2026_paper.html)) | 2026 | image/video/audio/talking-head, 1M+ | 1M+ | official release pending; an [unofficial HF copy](https://huggingface.co/datasets/JamalLee/Omni-Fake-SET) exists, not verified |
| [Community Forensics](https://arxiv.org/pdf/2411.04125) | CVPR 2025 | AI images from 4,803 generators | 2.7M | project page; eval set CC-BY-NC-SA |
| [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake/blob/dfcbc1e885bbe93fa4c204ac1fdd18d22de33f16/README.md) | 2025 | modern AI images | large | Hugging Face |
| [MLAAD](https://arxiv.org/html/2401.09512v6) | 2024-25 | TTS voices, 38 languages | 175k clips | open |
| [SpoofCeleb](https://arxiv.org/html/2409.17285v2) | 2025 | in-the-wild spoofed speech | 2.5M utterances | public |
| ASVspoof5 | 2024 | 32 generators | 1.2M utterances | registration |
| [SpeechFake](https://arxiv.org/html/2507.21463v1) | ACL 2025 | multilingual, recent generators | large | check page |

Held out for testing only (never train on): OmniFake-OOD, In-the-Wild, AIGIBench, CocoGlide, GenVideo, Deepfake-Eval-2024.

## 4. Other room for improvement (no big training)

- Grad-CAM face heatmap on Spatial (explainability) -- no training.
- Voice: per-second scores to highlight cloned segments -- no training.
- Lip-sync detector (e.g. AVH-Align) -- pretrained, test then add.
- Check GenD on the 16 local FF++ videos before adding it to the app (rule: must not lose ground on FF++).
- Small-image path for AI-image detection (<512 px).

## 5. Time estimates on this laptop (rough)

| Job | Time |
|---|---|
| Face-crop extraction from DF40 / Celeb-DF subset (~50k crops) | 3-5 h |
| Spatial fine-tune | 3-6 h |
| Frequency fine-tune | 2-4 h |
| GenD LayerNorm fine-tune | 4-8 h |
| Temporal fine-tune | 4-8 h |
| Voice fine-tune (MLAAD subset) | 4-8 h |
| Re-evaluation on all held-out benchmarks | 3-4 h |

## 6. Training-script review (2026-11-17) -- changes to make in the new retraining scripts (originals kept)

| Issue (current scripts) | Spatial | Frequency | Temporal | Fix |
|---|---|---|---|---|
| No compression / blur / resize augmentation (main cause of poor cross-dataset results in the literature) | yes | partial | yes | random JPEG q30-95, Gaussian blur, downscale-upscale, H.264 re-encode for video |
| Model chosen by val **loss** on same-dataset val only | yes | yes | yes | select by **AUC on a held-out other-dataset val** (e.g. Celeb-DF / DF40 unseen methods) |
| Only accuracy reported; no AUC / EER | yes | yes | yes | log AUC, EER, TPR at 5% FPR |
| Frame-level splits may leak identities/videos between train and val | check | check | check | split by source video / identity |
| No mixed precision (slower, less batch on 6 GB) | yes | yes | has AMP | `torch.amp.autocast` + GradScaler |
| StepLR schedule | yes | yes | warmup+cosine (good) | warmup + cosine everywhere |
| Python-level per-sample label mapping in the loop | yes | -- | -- | map once in the dataset |
| num_workers=8 (Windows memory pressure; caused job kills) | yes | -- | -- | 2-4 workers, pin_memory False |
| Single training source (FF++) | yes | yes | yes | FF++ + DF40 + Celeb-DF++ (+ own Dataset/Train) with per-source balancing |
| Fusion fitted on FF++ scores only | -- | -- | -- | refit on mixed held-out scores, add GenD input |

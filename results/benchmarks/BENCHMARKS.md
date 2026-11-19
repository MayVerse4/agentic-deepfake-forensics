
## TruFor localization on CocoGlide (512 real + 512 GLIDE-inpainted)
_2026-10-30T14:12:04 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **n_real**: 512
- **n_edited**: 512
- **image_auc**: 0.753
- **image_accuracy_at_0.5**: 0.6436
- **edited_caught_pct**: 35.4
- **real_flagged_pct**: 6.6
- **pixel_f1_at_0.5**: 0.3623
- **pixel_f1_best_threshold**: 0.6541
- **pixel_iou_at_0.5**: 0.2942
- **per_image_csv**: localization_eval.csv

## Community Forensics AI detector on random sample of CommunityForensics-Eval/CompEval (70x15 rows)
_2026-10-30T14:24:37 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **n_images**: 900
- **n_real**: 374
- **n_ai**: 526
- **generators_sampled**: 17
- **accuracy_at_0.5**: 0.8444
- **auc**: 0.9678
- **ai_detected_pct**: 73.8
- **real_flagged_pct**: 0.5
- **per_image_csv**: ai_detector_eval.csv

| group | n | detected % |
|---|---|---|
| MidjourneyV6_1 | 105 | 54.3 |
| Dalle2 | 60 | 80.0 |
| MidjourneyV5_2 | 59 | 93.2 |
| Firefly_Image3 | 45 | 28.9 |
| IdeogramV1 | 45 | 71.1 |
| DFGAN | 30 | 96.7 |
| Imagen3 | 30 | 83.3 |
| FLUX-schnell | 16 | 93.8 |
| Firefly_Image2 | 16 | 81.2 |
| Dalle3 | 15 | 100.0 |
| DeciDiffusionV2 | 15 | 100.0 |
| FLUX-dev | 15 | 93.3 |
| GALIP | 15 | 100.0 |
| Hourglass | 15 | 13.3 |
| IdeogramV2 | 15 | 66.7 |
| LCM_lora_ssd1b | 15 | 100.0 |
| stable_cascade | 15 | 100.0 |

## Separate vs combined on CocoGlide (held-out half, 5 splits, threshold for <=5% FPR on tuning half)
_2026-10-30T14:25:55 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **TruFor alone**: AUC 0.737, edits caught 29.0%, real flagged 5.0%
- **AI detector alone**: AUC 0.755, edits caught 36.8%, real flagged 4.9%
- **max(TruFor, AI)**: AUC 0.786, edits caught 43.0%, real flagged 4.9%
- **Combined (logistic)**: AUC 0.798, edits caught 44.5%, real flagged 4.6%
- **combiner_weights (TruFor, AI, bias)**: [3.504, 2.854, -1.485]

## mesorch localization on CocoGlide (512 real + 512 GLIDE-inpainted), same protocol as TruFor
_2026-10-30T14:48:53 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **image_auc**: 0.7427
- **edited_caught_pct**: 77.1
- **real_flagged_pct**: 47.3
- **pixel_f1_at_0.5**: 0.45
- **pixel_f1_best_threshold**: 0.6786
- **pixel_iou_at_0.5**: 0.3825
- **trufor_same_protocol**: {'image_auc': 0.753, 'edited_caught_pct': 35.4, 'real_flagged_pct': 6.6, 'pixel_f1_at_0.5': 0.3623, 'pixel_f1_best_threshold': 0.6541, 'pixel_iou_at_0.5': 0.2942}
- **per_image_csv**: mesorch_cocoglide_eval.csv

## All tools separate vs combined on CocoGlide (held-out half, 5 splits, <=5% FPR threshold)
_2026-10-30T14:49:38 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **TruFor alone**: AUC 0.737, edits caught 29.0%, real flagged 5.0%
- **Mesorch alone**: AUC 0.740, edits caught 23.9%, real flagged 4.5%
- **AI detector alone**: AUC 0.755, edits caught 36.8%, real flagged 4.9%
- **TruFor + Mesorch**: AUC 0.782, edits caught 38.2%, real flagged 5.6%
- **TruFor + AI detector**: AUC 0.798, edits caught 44.5%, real flagged 4.6%
- **Mesorch + AI detector**: AUC 0.804, edits caught 45.8%, real flagged 5.8%
- **TruFor + Mesorch + AI detector**: AUC 0.822, edits caught 50.6%, real flagged 5.9%
- **weights (TruFor, Mesorch, AI, bias)**: [2.879, 1.807, 2.686, -2.392]

## Eye-reflection physics check (corneal highlight IoU) on FFHQ real vs StyleGAN faces, 140k-Real-and-Fake-Faces test
_2026-10-30T15:09:44 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **n_images**: 600
- **image_size_px**: 256
- **usable**: 524
- **auc_on_usable**: 0.4525
- **mean_iou_real**: 0.042
- **mean_iou_fake**: 0.059
- **skipped**: {'no highlight in one or both eyes': 71, 'eyes too small (38px apart)': 1, 'eyes too small (18px apart)': 1, 'eyes too small (33px apart)': 1, 'eyes too small (20px apart)': 1, 'eyes too small (12px apart)': 1}
- **per_image_csv**: eye_reflection_eval.csv

## Eye-reflection physics check at 1024 px: FFHQ real vs StyleGAN2 faces
_2026-10-30T15:20:18 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **n_images**: 20
- **usable**: 17
- **auc_on_usable**: 0.2286
- **mean_iou_real**: 0.042
- **mean_iou_fake**: 0.205
- **skipped**: {'no highlight in one or both eyes': 3}
- **per_image_csv**: eye_reflection_hires_eval.csv

## Eye-reflection physics check at 1024 px: FFHQ real vs StyleGAN2 faces
_2026-10-30T15:40:31 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **n_images**: 521
- **usable**: 336
- **auc_on_usable**: 0.4597
- **mean_iou_real**: 0.156
- **mean_iou_fake**: 0.169
- **skipped**: {'no highlight in one or both eyes': 183, 'eyes too small (38px apart)': 1, 'eyes too small (8px apart)': 1}
- **per_image_csv**: eye_reflection_hires_eval.csv

## NSG-VD physics video detector (standard-Pika-d) on GenVideo shard 0 (10 generators)
_2026-10-30T18:17:13 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **n_real_test**: 136
- **n_ref_real**: 100
- **n_ai**: 587
- **auc**: 0.7907
- **ai_detected_at_5pct_fpr**: 43.6
- **per_generator (n, AUC, detected@5%FPR)**: {'COG': (60, 0.746, 40.0), 'DynamicCrafter': (62, 0.743, 35.5), 'Latte': (64, 0.931, 75.0), 'OpenSora': (66, 0.783, 43.9), 'Pika': (53, 0.924, 67.9), 'SEINE': (58, 0.777, 31.0), 'ST2V': (59, 0.633, 13.6), 'SVD': (61, 0.767, 36.1), 'VideoCrafter': (47, 0.85, 53.2), 'ZeroScope': (57, 0.77, 42.1)}
- **per_video_csv**: nsgvd_standard-Pika-d_eval.csv

## Videos: NSG-VD vs AI detector vs combined on GenVideo shard 0 (held-out half, 5 splits, <=5% FPR)
_2026-10-30T19:16:10 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **NSG-VD alone (physics)**: AUC 0.794, AI caught 44.7%, real flagged 6.7%
- **AI detector alone**: AUC 0.994, AI caught 95.7%, real flagged 6.2%
- **NSG-VD + AI detector**: AUC 0.989, AI caught 96.3%, real flagged 4.8%
- **per_generator_caught_pct (NSG, AI, combined)**: {'COG': [32.5, 88.9, 85.0], 'DynamicCrafter': [39.1, 100.0, 100.0], 'Latte': [76.1, 92.0, 92.0], 'OpenSora': [43.6, 100.0, 99.4], 'Pika': [67.7, 99.2, 98.5], 'SEINE': [29.5, 98.1, 98.7], 'ST2V': [17.6, 100.0, 100.0], 'SVD': [40.1, 100.0, 100.0], 'VideoCrafter': [57.2, 100.0, 100.0], 'ZeroScope': [42.8, 78.3, 90.9]}
- **combiner weights (log NSG, AI, bias)**: [3.522, 5.865, -2.061]

## NSG-VD physics video detector (standard-Pika-d) on GenVideo shard 0 (10 generators)_genvideo_norm
_2026-10-30T20:09:08 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **n_real_test**: 519
- **n_ref_real**: 100
- **n_ai**: 585
- **auc**: 0.5824
- **ai_detected_at_5pct_fpr**: 8.4
- **per_generator (n, AUC, detected@5%FPR)**: {'COG': (58, 0.575, 6.9), 'DynamicCrafter': (62, 0.607, 12.9), 'Latte': (64, 0.612, 9.4), 'OpenSora': (66, 0.502, 4.5), 'Pika': (53, 0.604, 9.4), 'SEINE': (58, 0.6, 13.8), 'ST2V': (59, 0.555, 3.4), 'SVD': (61, 0.585, 8.2), 'VideoCrafter': (47, 0.523, 6.4), 'ZeroScope': (57, 0.659, 8.8)}
- **per_video_csv**: nsgvd_standard-Pika-d_genvideo_norm_eval.csv

## Videos: NSG-VD vs AI detector vs combined on GenVideo shard 0 (held-out half, 5 splits, <=5% FPR)_genvideo_norm
_2026-10-30T20:13:03 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **NSG-VD alone (physics)**: AUC 0.589, AI caught 7.3%, real flagged 4.1%
- **AI detector alone**: AUC 0.979, AI caught 85.8%, real flagged 4.0%
- **NSG-VD + AI detector**: AUC 0.949, AI caught 81.6%, real flagged 3.7%
- **per_generator_caught_pct (NSG, AI, combined)**: {'COG': [6.7, 75.5, 74.5], 'DynamicCrafter': [9.4, 94.3, 91.8], 'Latte': [6.3, 78.1, 60.5], 'OpenSora': [4.5, 89.2, 84.2], 'Pika': [9.3, 72.3, 59.7], 'SEINE': [13.2, 88.7, 87.9], 'ST2V': [1.3, 93.5, 93.5], 'SVD': [8.6, 96.7, 93.5], 'VideoCrafter': [6.7, 100.0, 100.0], 'ZeroScope': [7.9, 67.8, 70.0]}
- **combiner weights (log NSG, AI, bias)**: [1.461, 8.335, -1.834]

## EXIF consistency checks (thumbnail, size, dates, quant tables, MakerNote) on 4 camera/phone JPEGs (ianare/exif-samples) + crops/edits/re-saves with original EXIF kept
_2026-10-30T20:57:14 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **originals_passed**: 4/4
- **resaved_same_picture_passed**: 4/4
- **cropped_caught**: 4/4
- **locally_edited_caught**: 4/4
- **note**: small test set; phones without an EXIF thumbnail (Nokia 8.3, Jolla) get no thumbnail check

## Invisible SD watermark decoder (invisible-watermark dwtDct, SD v1/v2 + SDXL messages)
_2026-10-30T20:57:14 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **clean_images_flagged**: 0/518 (512 CocoGlide real + 6 demo)
- **clean_bit_match**: median 0.51, max 0.69; threshold 0.80
- **sd1_watermark_found_untouched**: 33/50
- **sdxl_watermark_found_untouched**: 39/50
- **after_jpeg90_or_jpeg75_or_resize**: 0/50 (watermark destroyed)

## AI-inpainted edits: CocoGlide (512 real + 512 GLIDE-inpainted)
_2026-10-31T00:57:41 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **B-Free alone**: AUC 0.893, caught 64.4%, real flagged 4.4%
- **Community Forensics alone**: AUC 0.755, caught 36.8%, real flagged 4.9%
- **3 tools (current app)**: AUC 0.822, caught 50.6%, real flagged 5.9%
- **3 tools + B-Free**: AUC 0.886, caught 66.3%, real flagged 4.2%

## AI videos: GenVideo, format-normalized (519 real, 585 AI)
_2026-10-31T02:00:56 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **Community Forensics alone (current)**: AUC 0.979, caught 85.8%, real flagged 4.0%
- **B-Free alone**: AUC 0.894, caught 65.4%, real flagged 4.2%
- **Both combined**: AUC 0.947, caught 86.8%, real flagged 4.3%
- **per_group_caught_pct**: {'COG': [75.5, 40.8, 74.3], 'DynamicCrafter': [94.3, 79.3, 98.5], 'Latte': [78.1, 52.1, 78.2], 'OpenSora': [89.2, 76.4, 95.0], 'Pika': [72.3, 31.1, 62.6], 'SEINE': [88.7, 73.1, 93.8], 'ST2V': [93.5, 48.7, 94.2], 'SVD': [96.7, 91.5, 98.7], 'VideoCrafter': [100.0, 94.6, 100.0], 'ZeroScope': [67.8, 68.5, 69.2]}

## AI images: CommunityForensics-Eval sample (899 images, 17 generators)
_2026-10-31T02:18:14 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **Community Forensics alone (current)**: AUC 0.966, caught 87.6%, real flagged 6.1%
- **B-Free alone**: AUC 0.924, caught 71.6%, real flagged 4.7%
- **Both combined**: AUC 0.967, caught 89.8%, real flagged 4.7%
- **per_group_caught_pct**: {'DFGAN': [100.0, 86.6, 100.0], 'Dalle2': [97.0, 71.6, 95.7], 'Dalle3': [100.0, 100.0, 100.0], 'DeciDiffusionV2': [100.0, 100.0, 100.0], 'FLUX-dev': [95.0, 15.4, 95.0], 'FLUX-schnell': [100.0, 55.1, 100.0], 'Firefly_Image2': [94.9, 92.1, 94.9], 'Firefly_Image3': [71.2, 80.2, 90.4], 'GALIP': [100.0, 67.2, 100.0], 'Hourglass': [14.2, 0.0, 5.9], 'IdeogramV1': [89.0, 95.1, 96.8], 'IdeogramV2': [79.6, 80.4, 89.1], 'Imagen3': [92.0, 58.9, 89.0], 'LCM_lora_ssd1b': [100.0, 100.0, 100.0], 'MidjourneyV5_2': [95.1, 93.7, 97.0], 'MidjourneyV6_1': [79.6, 44.9, 78.8], 'stable_cascade': [100.0, 100.0, 100.0]}

## Voice-clone detectors on In-the-Wild (shard 0 sample: 750 real + 750 fake, first 6 s)
_2026-10-31T17:55:49 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **mo-thecreator/Deepfake-audio-detection**: AUC 0.947, EER 11.4%, caught 80.5% at 5% real flagged, acc@0.5 88.0%
- **Gustking/wav2vec2-large-xlsr-deepfake-audio-classification**: AUC 0.942, EER 13.1%, caught 80.5% at 5% real flagged, acc@0.5 82.6%
- **Bisher/wav2vec2_ASV_deepfake_audio_detection**: AUC 0.843, EER 22.1%, caught 44.5% at 5% real flagged, acc@0.5 52.9%
- **Hemgg/Deepfake-audio-detection**: AUC 0.793, EER 26.7%, caught 21.9% at 5% real flagged, acc@0.5 74.1%
- **Vansh180/deepfake-audio-wav2vec2**: AUC 0.595, EER 45.6%, caught 5.3% at 5% real flagged, acc@0.5 51.1%
- **MelodyMachine/Deepfake-audio-detection-V2**: AUC 0.417, EER 53.3%, caught 0.0% at 5% real flagged, acc@0.5 42.5%
- **per_clip_csv**: voice_eval.csv

## AI images: AIGIBench independent benchmark (998 images, 10 subsets)
_2026-10-31T18:26:56 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **Community Forensics alone**: AUC 0.965, caught 79.8%, real flagged 3.7%
- **B-Free alone**: AUC 0.958, caught 81.3%, real flagged 5.1%
- **Both combined**: AUC 0.981, caught 88.0%, real flagged 3.9%
- **per_group_caught_pct**: {'CommunityAI': [64.1, 80.0, 82.4], 'DALLE-3': [99.2, 100.0, 100.0], 'FLUX1-dev': [76.6, 36.1, 61.6], 'Imagen3': [47.5, 68.9, 65.6], 'InstantID': [75.5, 98.5, 100.0], 'Midjourney': [82.1, 85.5, 87.8], 'SD3': [89.9, 100.0, 100.0], 'SDXL': [100.0, 100.0, 100.0], 'SocialRF': [64.7, 81.2, 84.4], 'StyleGAN-XL': [100.0, 66.5, 100.0]}

## AIGIBench: CF, B-Free, SPAI, GAPL alone and combined (998 images; 5 splits; <=5% real flagged)
_2026-11-16T20:06:08 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **Community Forensics**: AUC 0.965, caught 79.8%, real flagged 3.7%, acc 88.1%, F1 86.9%
- **B-Free**: AUC 0.958, caught 81.3%, real flagged 5.1%, acc 88.1%, F1 87.2%
- **SPAI (CVPR 2025)**: AUC 0.759, caught 33.2%, real flagged 4.2%, acc 64.4%, F1 48.2%
- **GAPL (CVPR 2026)**: AUC 0.968, caught 84.1%, real flagged 4.3%, acc 89.9%, F1 89.3%
- **CF + B-Free (app now)**: AUC 0.981, caught 88.0%, real flagged 3.9%, acc 92.1%, F1 91.8%
- **CF + B-Free + SPAI**: AUC 0.975, caught 86.6%, real flagged 3.8%, acc 91.4%, F1 91.0%
- **CF + B-Free + GAPL**: AUC 0.990, caught 95.2%, real flagged 4.4%, acc 95.4%, F1 95.4%
- **All four**: AUC 0.989, caught 94.3%, real flagged 4.2%, acc 95.0%, F1 95.0%
- **per_subset_caught_at_0.5 (CF, BFree, SPAI, GAPL)**: {'CommunityAI': [58.0, 76.0, 76.0, 70.0], 'DALLE-3': [96.0, 98.0, 86.0, 88.0], 'FLUX1-dev': [64.0, 34.0, 66.0, 92.0], 'Imagen3': [28.0, 56.0, 50.0, 90.0], 'InstantID': [64.0, 98.0, 96.0, 98.0], 'Midjourney': [62.0, 82.0, 90.0, 94.0], 'SD3': [78.0, 100.0, 98.0, 100.0], 'SDXL': [100.0, 100.0, 100.0, 100.0], 'SocialRF': [58.0, 80.0, 62.0, 88.0], 'StyleGAN-XL': [100.0, 48.0, 100.0, 100.0]}
- **notes**: SPAI official CLI with --resize-to 1024 (6 GB GPU); GAPL official checkpoint, all 541 tensors loaded

## OmniFake-OOD: Voice: REAL vs fake speech
_2026-11-16T20:49:54 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **mo-thecreator**: AUC(app) 0.644, AUC 0.654, caught 0.0%, flagged 0.0%, acc 44.4%
- **Gustking**: AUC(app) 0.978, AUC 0.977, caught 90.5%, flagged 5.3%, acc 92.4%
- **mo + Gustking (app)**: AUC(app) 0.889, AUC 0.971, caught 88.7%, flagged 5.2%, acc 91.5%

## OmniFake-OOD: Video: REAL vs fake video
_2026-11-16T20:53:52 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **Community Forensics, 8 frames (app)**: AUC(app) 0.766, AUC 0.769, caught 49.4%, flagged 5.4%, acc 64.5%

## OmniFake-OOD: AI image: REAL vs FULL_SYNTHETIC
_2026-11-16T21:03:53 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **Community Forensics**: AUC(app) 0.870, AUC 0.878, caught 52.8%, flagged 5.2%, acc 74.5%
- **B-Free**: AUC(app) 0.526, AUC 0.544, caught 7.0%, flagged 5.0%, acc 52.3%
- **GAPL**: AUC(app) 0.850, AUC 0.856, caught 51.3%, flagged 4.2%, acc 74.2%
- **CF + B-Free (app now)**: AUC(app) 0.756, AUC 0.759, caught 51.3%, flagged 4.5%, acc 74.1%
- **CF + B-Free + GAPL (proposed)**: AUC(app) 0.844, AUC 0.860, caught 60.9%, flagged 3.5%, acc 79.2%

## OmniFake-OOD: Edited image: REAL vs TAMPERED
_2026-11-16T21:03:53 on NVIDIA GeForce RTX 4050 Laptop GPU_

- **TruFor**: AUC(app) 0.526, AUC 0.537, caught 7.8%, flagged 6.6%, acc 52.5%
- **Mesorch**: AUC(app) 0.507, AUC 0.497, caught 7.2%, flagged 3.6%, acc 53.8%
- **4 tools (app now)**: AUC(app) 0.622, AUC 0.634, caught 12.3%, flagged 5.1%, acc 55.6%
- **4 tools + GAPL**: AUC(app) nan, AUC 0.722, caught 25.5%, flagged 7.0%, acc 60.9%

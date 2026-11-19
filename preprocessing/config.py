class Config:
    # ========================
    # PATHS
    # ========================
    OUTPUT_DIR = "processed_data"

    # ========================
    # IMAGE / FRAME SETTINGS
    # ========================
    IMAGE_SIZE = 224
    FRAME_RATE = 5  # frames per second

    # ========================
    # STANDARDIZATION
    # ========================
    # Applied after 0-1 normalization, per-channel (RGB order).
    # ImageNet stats by default -- confirm with whoever builds the Spatial
    # model that this matches their backbone (pretrained vs from-scratch).
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # ========================
    # SEQUENCE SETTINGS
    # ========================
    SEQUENCE_LENGTH = 16

    # ========================
    # FREQUENCY SETTINGS
    # ========================
    FFT_LOG_SCALE = True
    FFT_NORMALIZE = True

    # Frequency MODEL input (HFRI). Must match what the checkpoint was trained
    # with -- see models/frequency/frequency_model.compute_high_freq_image_gpu.
    FREQ_MASK_RADIUS = 0.25
    FREQ_NORM = "std"
    FREQUENCY_CHECKPOINT = "frequency_resnet18_std.pth"

    # 3-input (no frequency) or 4-input fusion checkpoint; FusionModel.load
    # detects which from the weights.
    FUSION_CHECKPOINT = "fusion_model_4input.pth"

    # Temporal-frequency (novelty addition): per-pixel 1D FFT along the
    # time axis across a sequence window. Requires >=2 frames per sequence
    # to produce a meaningful spectrum.
    ENABLE_TEMPORAL_FREQUENCY = True

    # ========================
    # FACE FILTERING
    # ========================
    MIN_FACE_SIZE = 50  # pixels
    FACE_CONFIDENCE_THRESHOLD = 0.5

    # ========================
    # BLUR DETECTION
    # ========================
    BLUR_THRESHOLD_EXTREME = 5     # only garbage frames get discarded
    BLUR_THRESHOLD_LOW = 80        # below this = classified "medium"

    # ========================
    # TRACKING
    # ========================
    USE_DEEPSORT = True
    MAX_TRACK_LOST = 5
    DEEPSORT_MAX_AGE = 15
    DEEPSORT_N_INIT = 2

    # ========================
    # STORAGE
    # ========================
    SAVE_FORMAT = {
        "image": "jpg",
        "array": "npy",
        "audio": "wav"
    }

    # ========================
    # PARALLEL PROCESSING
    # ========================
    NUM_WORKERS = 4  # adjust based on CPU

    # ========================
    # LIMITS (safety)
    # ========================
    MAX_FRAMES = 500  # prevent huge videos
    # RetinaFace runs on CPU (TF 2.15 has no RTX 50-series support); detecting on
    # a downscaled copy is ~7x faster on 1080p (1.97 -> 0.27 s/frame, boxes within
    # a few px). Boxes are mapped back, and crops
    # are still cut from the full-resolution frame. None = full resolution.
    DETECTION_MAX_SIDE = 640

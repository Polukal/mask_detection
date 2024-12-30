class Config:
    # Dataset parameters
    DATASET_PATH = "face-mask-detection"
    IMAGE_SIZE = (416, 416)  # Standard size for YOLO
    BATCH_SIZE = 32
    NUM_CLASSES = 3  # with_mask, without_mask, mask_worn_incorrect

    # Training parameters
    NUM_EPOCHS = 80
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.8

    # Model parameters
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4

    # Classes
    CLASSES = ['with_mask', 'without_mask', 'mask_worn_incorrect']

    # Paths
    MODEL_SAVE_PATH = "saved_models"
    LOGS_PATH = "logs"
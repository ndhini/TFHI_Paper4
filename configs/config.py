# ==========================================================
# FAST / FULL RESEARCH CONFIGURATION
# ==========================================================

FAST_MODE = False

# ==========================================================

if FAST_MODE:

    # Fast Research Mode (20–30 minutes)

    SSL_EPOCHS = 3
    TRAIN_EPOCHS = 3

    BATCH_SIZE = 128

    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4

    USE_SUBSET = True

    TRAIN_SUBSET = 10000
    VAL_SUBSET = 2000

else:

    # Full Training Mode (12+ hours)

    SSL_EPOCHS = 20
    TRAIN_EPOCHS = 100

    BATCH_SIZE = 128

    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4

    USE_SUBSET = False

    TRAIN_SUBSET = None
    VAL_SUBSET = None
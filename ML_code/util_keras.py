import os
from keras.callbacks import ModelCheckpoint

# This file is for utility functions that require keras library

def create_checkpoint_callback(subdirectory, filename_template):
    checkpoint = ModelCheckpoint(
        os.path.join(subdirectory, filename_template),
        save_weights_only=True,
        period=1
    )
    return checkpoint
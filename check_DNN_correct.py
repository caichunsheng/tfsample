import tensorflow as tf

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)  # Virtual devices must be set at program startup
except Exception as e:
    print(f"An error occurred while initializing TensorFlow: {e}")

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Num GPUs Available: {len(gpus)}")
    print("TensorFlow is using: GPU")
else:
    print("Num GPUs Available: 0")
    print("TensorFlow is using: CPU")

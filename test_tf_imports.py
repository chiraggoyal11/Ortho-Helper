import sys
import os

print(f"Python executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow location: {os.path.dirname(tf.__file__)}")
except ImportError as e:
    print(f"Failed to import tensorflow: {e}")

try:
    from tensorflow import keras
    print(f"from tensorflow import keras worked. Keras version: {keras.__version__}")
except ImportError as e:
    print(f"from tensorflow import keras FAILED: {e}")

try:
    import tensorflow.keras as tf_keras
    print(f"import tensorflow.keras worked. Version: {tf_keras.__version__}")
except ImportError as e:
    print(f"import tensorflow.keras FAILED: {e}")

try:
    import keras
    print(f"import keras worked. Version: {keras.__version__}")
    print(f"Keras location: {os.path.dirname(keras.__file__)}")
except ImportError as e:
    print(f"import keras FAILED: {e}")

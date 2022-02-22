from tensorflow import keras
loaded = keras.models.load_model("model")
print("Loaded model:", loaded)
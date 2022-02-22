from tensorflow.keras import models  # 导入TF子库
network = models.load_model('model.h5')
print("loaded model !")
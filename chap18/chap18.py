from tensorflow import keras

callbacks = [
    # 当验证集上损失连续两个周期都低于0.01时，停止训练
    keras.callback.EarlyStopping(
        monitor = "val_loss",
        min_delta = 1e-2,
        patience = 2,
        verbose = 1,
    )
]

"""
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=16,
    callbacks=callbacks,
    validation_split=0.2,
)
"""

# 下面是一个简单的示例，在训练期间保存每个批次的损失值列表
class lossLog(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))


# 为模型设置检查点
callbacks = [
    keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath="mymodel_{epoch}",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
    )
]

"""
model.fit(
    x_train, y_train, epochs=2, batch_size=64, callbacks=callbacks, validation_split=0.2
)
"""


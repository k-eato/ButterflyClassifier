import os
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf

# Define variables
train_data_dir = pathlib.Path("../../data/train")
val_data_dir = pathlib.Path("../../data/valid")
img_shape = [224, 224, 3]
img_size = 224
batch_size = 64
base_learning_rate = 0.0005
initial_epochs = 20
total_epochs = initial_epochs + 10
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Import datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    seed=123,
    label_mode="categorical",
    image_size=(img_size, img_size),
    batch_size=batch_size,
)
validation_ds = tf.keras.utils.image_dataset_from_directory(
    val_data_dir,
    seed=123,
    label_mode="categorical",
    image_size=(img_size, img_size),
    batch_size=batch_size,
)

# Build model from pretrained MobileNet
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=img_shape, include_top=False, weights="imagenet"
)
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)
feature_batch_average = global_average_layer(feature_batch)
dense_layer = tf.keras.layers.Dense(150)
prediction_layer = tf.keras.layers.Dense(75)
dense_batch = dense_layer(feature_batch_average)
prediction_batch = prediction_layer(dense_batch)
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = dense_layer(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Setup and run training
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)
history = model.fit(
    train_ds,
    epochs=initial_epochs,
    validation_data=validation_ds,
    callbacks=[cp_callback],
)
for layer in base_model.layers[100:]:
    layer.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate / 10),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_ds,
    callbacks=[cp_callback],
)

# Plot training results
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
acc += history_fine.history["accuracy"]
val_acc += history_fine.history["val_accuracy"]
loss += history_fine.history["loss"]
val_loss += history_fine.history["val_loss"]
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.ylim([0.8, 1])
plt.plot(
    [initial_epochs - 1, initial_epochs - 1],
    plt.ylim(),
    label="Start Fine Tuning",
)
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.ylim([0, 1.0])
plt.plot(
    [initial_epochs - 1, initial_epochs - 1],
    plt.ylim(),
    label="Start Fine Tuning",
)
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import warnings

# Suppress TFLite deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.lite")

# Load sample dataset (using CIFAR10 as simulation for recyclable items)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Reduce dataset to binary classification (simulate recyclable vs non-recyclable)
recyclable_classes = [1, 5]  # Example: automobiles & birds = recyclable simulation
mask_train = np.isin(y_train, recyclable_classes)
mask_test = np.isin(y_test, recyclable_classes)

x_train = x_train[mask_train.flatten()]
y_train = np.isin(y_train[mask_train.flatten()], recyclable_classes).astype(int)
x_test = x_test[mask_test.flatten()]
y_test = np.isin(y_test[mask_test.flatten()], recyclable_classes).astype(int)

# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# Lightweight CNN Model
model = models.Sequential([
    layers.Input(shape=(32,32,3)),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

# Convert Model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("recycle_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Model converted to TensorFlow Lite and saved as recycle_model.tflite")

# Test TFLite Model
interpreter = tf.lite.Interpreter(model_path="recycle_model.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

sample = x_test[0:1].astype('float32')
interpreter.set_tensor(input_index, sample)
interpreter.invoke()

prediction = interpreter.get_tensor(output_index)
print("Prediction:", prediction)

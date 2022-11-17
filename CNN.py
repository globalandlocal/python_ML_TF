import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

RMS = tf.keras.optimizers.RMSprop
IDG = tf.keras.preprocessing.image.ImageDataGenerator
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss="binary_crossentropy", metrics="acc")
training = "./Cats&Dogs/Training/"
testing = "./Cats&Dogs/Testing/"
DG = IDG(rescale=1.0/255)
training_D = DG.flow_from_directory(training, batch_size=100, class_mode="binary", target_size=(150, 150))
testing_D = DG.flow_from_directory(testing, batch_size=100, class_mode="binary", target_size=(150, 150))
history = model.fit(training_D, epochs=4, validation_data=testing_D)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', "Training Acc-red")
plt.plot(epochs, val_acc, 'b', "Validation Acc-blue")
plt.title('Training and validation acc')
plt.figure()
plt.plot(epochs, loss, 'r', "Training Loss-red")
plt.plot(epochs, val_loss, 'b', "Validation Loss-blue")
plt.figure()

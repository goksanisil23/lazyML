import numpy as np
import tensorflow as tf


# Define the PointNet architecture
def pointnet_segmentation(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Input transformation network
    x = tf.keras.layers.Conv1D(64, 1, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Feature transformation network
    x = tf.keras.layers.Conv1D(64, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(1024, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    # Fully connected layers
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model


# Load pre-trained weights
model = pointnet_segmentation(input_shape=(2048, 3), num_classes=10)
model.load_weights("pointnet_segmentation_weights.h5")

# Example usage on arbitrary point cloud data
point_cloud = np.random.rand(2048, 3)  # Replace with your actual point cloud data

# Preprocess the point cloud (e.g., normalize, scale, etc.)
normalized_pc = (point_cloud - np.mean(point_cloud, axis=0)) / np.std(
    point_cloud, axis=0
)

# Make predictions
predictions = model.predict(np.expand_dims(normalized_pc, axis=0))

# Get the predicted segmentation labels
segmentation_labels = np.argmax(predictions, axis=-1)

print(segmentation_labels)

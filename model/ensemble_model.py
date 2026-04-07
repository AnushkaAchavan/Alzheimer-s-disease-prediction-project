import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50, DenseNet121
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Input,
    Dropout, BatchNormalization, Concatenate
)
from tensorflow.keras.models import Model


def build_ensemble_model(num_classes=3, input_shape=(224, 224, 3)):

    input_layer = Input(shape=input_shape, name="shared_input")

    # -------------------- EfficientNet --------------------
    effnet = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    effnet.trainable = False
    x1 = effnet(input_layer, training=False)
    x1 = GlobalAveragePooling2D()(x1)

    # -------------------- ResNet --------------------
    resnet = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    resnet.trainable = False
    x2 = resnet(input_layer, training=False)
    x2 = GlobalAveragePooling2D()(x2)

    # -------------------- DenseNet --------------------
    densenet = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    densenet.trainable = False
    x3 = densenet(input_layer, training=False)
    x3 = GlobalAveragePooling2D()(x3)

    # -------------------- Combine (Ensemble) --------------------
    combined = Concatenate(name="concat")([x1, x2, x3])

    # -------------------- Classification Head --------------------
    x = Dense(512, activation="relu")(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    output = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=input_layer, outputs=output, name="ensemble_model")

    return model
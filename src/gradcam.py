import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# EfficientNetB0's final conv/activation layer before GlobalAveragePooling2D.
# This is the standard target layer for Grad-CAM on EfficientNet backbones.
LAST_CONV_LAYER_NAME = "top_activation"


def get_base_model(model, last_conv_layer_name=LAST_CONV_LAYER_NAME):
    """
    Locate the EfficientNetB0 base model nested inside the outer
    classification model, regardless of its auto-assigned layer name
    (e.g. 'efficientnetb0' vs 'efficientnetb0_1').
    """
    for layer in model.layers:
        if hasattr(layer, "get_layer"):
            try:
                layer.get_layer(last_conv_layer_name)
                return layer
            except ValueError:
                continue
    raise ValueError(
        f"Could not find a nested base model containing layer "
        f"'{last_conv_layer_name}'. Check LAST_CONV_LAYER_NAME."
    )


def make_gradcam_heatmap(img_array, model, base_model=None,
                          last_conv_layer_name=LAST_CONV_LAYER_NAME,
                          pred_index=None):
    if base_model is None:
        base_model = get_base_model(model, last_conv_layer_name)

    conv_layer_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(last_conv_layer_name).output
    )

    # Everything in `model` that comes after the base_model layer --
    # the classification head (GAP, BatchNorm, Dense, Dropout, ...).
    base_index = model.layers.index(base_model)
    head_layers = model.layers[base_index + 1:]

    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        conv_output = conv_layer_model(img_tensor, training=False)
        tape.watch(conv_output)

        x = conv_output
        for layer in head_layers:
            x = layer(x, training=False)
        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        raise RuntimeError(
            "Gradient computation returned None -- check that "
            "last_conv_layer_name is correct and differentiable."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_image, heatmap, alpha=0.45, image_size=(224, 224)):
    """
    original_image: PIL Image (any mode/size -- will be RGB + resized).
    heatmap: 2D numpy array from make_gradcam_heatmap, values in [0, 1].
    alpha: heatmap opacity over the original image.

    Returns a PIL Image with the Grad-CAM heatmap overlaid.
    """
    img = np.array(original_image.convert("RGB").resize(image_size))

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlaid = np.uint8(heatmap_color * alpha + img * (1 - alpha))
    return Image.fromarray(overlaid)
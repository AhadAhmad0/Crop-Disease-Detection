import tensorflow as tf
import os
import numpy as np

print(f"TensorFlow version: {tf.__version__}")

# Load the keras model
print("Loading model...")
model = tf.keras.models.load_model("model/crop_disease_model.keras")
print("Model loaded successfully")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# Export to SavedModel format
save_path = "model/crop_disease_savedmodel"
os.makedirs("model", exist_ok=True)

print(f"Exporting to SavedModel format...")
model.export(save_path)
print(f"Export complete")

# Verify it loads correctly
print("Verifying SavedModel...")
loaded = tf.saved_model.load(save_path)
infer = loaded.signatures["serving_default"]

# Test with dummy input
test_input = np.random.randint(0, 255, (1, 224, 224, 3)).astype(np.float32)
result = infer(tf.constant(test_input))
output_key = list(result.keys())[0]
predictions = result[output_key].numpy()

print(f"Inference test passed")
print(f"Output shape: {predictions.shape}")
print(f"Output key: {output_key}")
print(f"Predictions sum: {predictions.sum():.4f} (should be ~1.0)")
print(f"\nSavedModel files:")
for root, dirs, files in os.walk(save_path):
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath) / (1024*1024)
        print(f"  {filepath} — {size:.2f} MB")
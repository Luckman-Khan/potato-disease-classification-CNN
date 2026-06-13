import tensorflow as tf
import keras
import os

# Define paths
model_path = r'c:\Projects\potato-disease-classification-CNN\Models\1.keras'
export_path = r'c:\Projects\potato-disease-classification-CNN\Models\potatoes_model\1'

print(f"Loading model using keras.models.load_model...")
# Load the model
try:
    model = keras.models.load_model(model_path, compile=False)
    
    print(f"Exporting model to {export_path}...")
    # Ensure directory exists
    os.makedirs(export_path, exist_ok=True)

    # Save in SavedModel format
    model.save(export_path)
    print("Conversion complete!")
except Exception as e:
    print(f"Error during conversion: {e}")

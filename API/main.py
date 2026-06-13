from fastapi import FastAPI,File , UploadFile
import numpy as np
from PIL import Image
import io
import uvicorn
import tensorflow as tf

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"
MODEL = tf.keras.models.load_model("../Models/1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(io.BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }





if __name__ == '__main__':
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)

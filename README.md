This project explores image classification of potato leaves using a Convolutional Neural Network (CNN).
The goal is to detect whether a potato plant is healthy or affected by early blight or late blight, based on the PlantVillage dataset.

## Dataset

Source: PlantVillage Dataset (Kaggle)

Classes:

- Early Blight
- Late Blight
- Healthy

Images are organized into training, validation, and test sets.

## Model

- Framework: TensorFlow / Keras
- Architecture: Convolutional Neural Network (CNN)
- Training: 50 epochs

## Streamlit Frontend

Install dependencies and run:

```bash
pip install -r Requirements.txt
streamlit run app.py
```

The app loads the exported TensorFlow SavedModel from
`Models/potato_model/1`. Users can upload an image, take a photo with their
camera, or try bundled demo images before viewing the predicted class and
confidence score.

The app lets a user upload a potato leaf image and predicts one of:

- `Potato___Early_blight`
- `Potato___Late_blight`
- `Potato___healthy`

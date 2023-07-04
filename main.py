from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3005",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = load_model("tomatoes2.h5")

CLASS_NAMES = ["Others", "Tomato_healthy", "Tomato_mosaic_virus"]


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))
    image = image.convert("RGB")  # Convert to RGB if the input image has a different mode
    image = np.array(image)
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    #file_extension = file.filename.split(".")[-1]
    image = read_file_as_image(await file.read())

    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        "class": predicted_class,
        "confidence": float(confidence),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

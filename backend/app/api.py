from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import functions
from PIL import Image

app = FastAPI()

origins = ["http://localhost:3000", "localhost:3000"]

prediction = [
    {
        "butterfly": "",
    }
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Read in image and receive prediction
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    image = np.array(Image.open(BytesIO((await file.read()))))
    prediction[0]["butterfly"] = functions.inference(image)


# Return prediction
@app.get("/result")
async def get_pred():
    return {"data": prediction[0]["butterfly"]}

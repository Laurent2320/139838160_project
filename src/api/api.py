from fastapi import FastAPI
from fastapi.responses import JSONResponse
import tensorflow as tf
from pydantic import BaseModel

import requests
import io
import zipfile
import pandas as pd

from google.cloud import storage

def load_model_from_gcs(bucket_name, model_file_name):
    try:
        # Initialize GCP Storage Client
        client = storage.Client.from_service_account_json('m-412710-831f51af70d6.json')

        # Specify GCP Storage Bucket and Model File Name
        model_blob_path = f"gs://{bucket_name}/{model_file_name}"

        # Download Model from GCP Storage
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_file_name)
        blob.download_to_filename(model_file_name)

        # Load the Model
        loaded_model = load_model(model_file_name)

        return loaded_model

    except Exception as e:
        print(f"Error: {e}")
        return None


bucket_name = "139m_model"
model_file_name = "test_valid_test_512_64_8_0.2.h5"

model = load_model_from_gcs(bucket_name, model_file_name)

app = FastAPI()

class Item(BaseModel):
    data: list

class InputData(BaseModel):
    num_results: int
    cost_euros: float

class OutputData(BaseModel):
    list_1: list

@app.post("/predict")
def predict(item: Item):
    try:
        # Perform any necessary data preprocessing on item.data
        # Make predictions using your loaded model
        predictions = model.predict(tf.convert_to_tensor([item.data]))

        # Return predictions as JSON
        return JSONResponse(content={"predictions": predictions.tolist()})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/generate_numbers", response_model=OutputData)
def generate_numbers(data: InputData):
    
    predictions = model.predict(tf.convert_to_tensor([item.data]))

    return OutputData(list_1=predictions )
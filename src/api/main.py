import os
import io
import zipfile
from typing import List

import pandas as pd
import numpy as np

import requests
from fastapi import FastAPI, HTTPException, staticfiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from keras.models import load_model
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

from module_lib import draws
from google.cloud import storage


path_key_file = "src/api/m-412710-b2b84652d635.json"
path_data_folder = "src/api/static_data"

# path_key_file = "m-412710-b2b84652d635.json"
# path_data_folder = "static_data"

def download_model_from_gcs(bucket_name, model_file_name):
    try:
        print(os.getcwd())
        print(path_key_file)
        # Initialize GCP Storage Client
        client = storage.Client.from_service_account_json(path_key_file)

        # Specify GCP Storage Bucket and Model File Name
        # model_blob_path = f"gs://{bucket_name}/{model_file_name}"

        # Download Model from GCP Storage
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_file_name)
        path_model_file = f"{path_data_folder}/{model_file_name}"
        blob.download_to_filename(path_model_file)
        
        print(path_model_file)
        return path_model_file

    except Exception as e:
        raise Exception(e)



app = FastAPI()

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",  # Update with the actual origin of your frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # You can specify specific HTTP methods (e.g., ["GET", "POST"])
    allow_headers=["*"],  # You can specify specific headers if needed
)

# app.mount("/static_data", staticfiles.StaticFiles(directory="static_data"), name="static")

# def load_files_on_startup():
#     # static_folder_path = "static_data"
#     # bucket_name = "139m_model"
#     # model_file_name = "test_valid_test_512_64_12_0.2.h5"
#     # download_model_from_gcs(bucket_name,model_file_name)
#   return 0

# app.add_event_handler("startup", load_files_on_startup)



class InputData(BaseModel):
    number_of_grid: int

class OutputData(BaseModel):
    list_1: List[List[int]]

@app.post("/generate_grids", response_model=OutputData)
def generate_grids(data: InputData):
    try:
        bucket_name = "139m_model"
        model_file_name = "test_valid_test_512_64_12_0.2.h5"
        path_model_file = download_model_from_gcs(bucket_name,model_file_name)
      
        # path_model_file = f"{path_data_folder}/test_valid_test_512_64_12_0.2.h5"
        model = load_model(path_model_file)
        
        if model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded.")
        
        print(model.summary())
        
        # Assuming draws function returns a list of floats
        selected_predictions = draws(data.number_of_grid, 12, model)

        # Convert the floats to integers
        selected_predictions_int = [[int(value) for value in sublist] for sublist in selected_predictions]

        return OutputData(list_1=selected_predictions_int)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    # Run the application on localhost with port 8000 using Uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')




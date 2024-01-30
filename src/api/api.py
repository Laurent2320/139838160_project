from fastapi import FastAPI
from fastapi.responses import JSONResponse
import tensorflow as tf
from pydantic import BaseModel

app = FastAPI()

# Initialize GCP Storage Client
storage_client = storage.Client()

# Specify GCP Storage Bucket and Model File Name
bucket_name = "139m"
model_file_name = "139m_model"
model_blob_path = f"gs://{bucket_name}/{model_file_name}"

# Download Model from GCP Storage
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(model_file_name)
blob.download_to_filename(model_file_name)

# Load the Model during Startup
model = tf.keras.models.load_model(model_file_name)


# URL of the ZIP file
url = "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_202002.zip"
# Send a GET request to the URL
response = requests.get(url)
# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Open a file-like object
    zip_file = io.BytesIO(response.content)
    # Extract the contents of the ZIP file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        # Find the CSV file within the ZIP
        csv_file_name = zip_ref.namelist()[0]
        # Read the CSV file into a DataFrame df_scraped
        with zip_ref.open(csv_file_name) as csv_file:
            df_scraped = pd.read_csv(csv_file, sep=';', encoding='ISO-8859-1')



class Item(BaseModel):
    data: list

class InputData(BaseModel):
    num_results: int
    cost_euros: float

class OutputData(BaseModel):
    list_1: list
    list_2: list
    list_3: list

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
    
    #scrap
    #load
    #predict
    #generate output

    return OutputData(list_1=list_1, list_2=list_2, )
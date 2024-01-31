# App.py
# loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import requests
import zipfile
import io

# frequency of a number in the last window rows
def count_frequency(dataframe, windows):
    df_ = dataframe.copy()
    for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']:
        df_[f'freq_{col}'] = df_.apply(lambda row: sum(df_[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].iloc[max(0, row.name - windows):row.name].values.flatten() == row[col]), axis=1)
    for col in ['etoile_1', 'etoile_2']:
        df_[f'freq_{col}'] = df_.apply(lambda row: sum(df_[['etoile_1', 'etoile_2']].iloc[max(0, row.name - windows):row.name].values.flatten() == row[col]), axis=1)
    return df_

# Compute squart difference between numbers
def quadra_dif(data):
  df_ = data.copy()
  columns_to_diff = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']
  columns_s_to_diff = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']
  for i in range(0, df_.shape[0]):
    df_['sum_diff_r2'] = ((df_[columns_to_diff].diff(axis=1) ** 2).sum(axis=1)).astype(int)
    df_['sum_diff_s_r2'] = ((df_[columns_s_to_diff].diff(axis=1) ** 2).sum(axis=1)).astype(int)
  return df_

# How long the number didn't not appear
def no_star(data):
  df_ = data.copy()
  for num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    mask = ((df_['etoile_1'] == num) | (df_['etoile_2'] == num))
    df_[f'no_s{num}'] = (~mask).groupby((mask).cumsum()).cumsum()
    # df_[f'no_{num}'] = df_[f'no_{num}'].shift(fill_value=0).astype(int)
  return df_

def no_ball(data):
  df_ = data.copy()
  for num in range(51):
    mask = ((df_['boule_1'] == num) | (df_['boule_2'] == num) | (df_['boule_3'] == num) | (df_['boule_4'] == num) | (df_['boule_5'] == num))
    df_[f'no_b{num}'] = (~mask).groupby((mask).cumsum()).cumsum()
  return df_

# Function to create sequences for X and y depending on the windows value
def create_sequences(data, length):
    X,Y = [],[]
    for i in range(len(data)-length):
        X.append(data.iloc[i+1:i+length+1, 0:data.shape[1]].values)
    for i in range(len(data)-length):
      Y.append(data.iloc[i+length, 0:7])
    return np.array(X),np.array(Y)

def post_traitement(predic):
  array = np.clip(np.round(predic[0]).astype(int), 1, 50)
  # Find indices of duplicate values for balls
  unique_values, counts = np.unique(array[:5], return_counts=True)
  duplicates = unique_values[counts > 1]
  # Replace duplicates in balls
  for value in duplicates:
    indices = np.where(array[:5] == value)[0]
    val0 = (predic[0][indices[0]] - array[indices[0]])**2
    val1 = (predic[0][indices[1]] - array[indices[1]])**2
    if val0 < val1:
        array[indices[1]] += 1 if (predic[0][indices[1]] > array[indices[1]] and ((array[indices[1]] + 1) not in array[:5])) else -1
    else:
        array[indices[0]] += 1 if (predic[0][indices[0]] > array[indices[0]] and ((array[indices[0]] + 1) not in array[:5])) else -1

  # Find indices of duplicate values for stars
  unique_star, counts_s = np.unique(array[-2:], return_counts=True)
  duplicates_star = any(counts_s > 1)
  if duplicates_star:
    star_indices = np.where(np.isin(array[-2:], unique_star[counts_s > 1]))[0]
    star0 = (predic[0][star_indices[0]] - array[star_indices[0]])**2
    star1 = (predic[0][star_indices[1]] - array[star_indices[1]])**2

    # Update array based on minimizing squared differences
    if star0 < star1:
        array[star_indices[1]+5] += 1 if predic[0][1] > array[star_indices[1]+5] else -1
    else:
        array[star_indices[0]+5] += 1 if predic[0][0] > array[star_indices[0]+5] else -1
  return array.tolist()


# Scrapping file results from FDJ URL
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

# Load the model from the H5 file
model = load_model('test_valid_test_512_64_12_0.2.h5')

# Define the sequence length
windows = 12

# Building Features for predictions from df_scraped
df_ligth = df_scraped[['boule_1','boule_2','boule_3','boule_4','boule_5','etoile_1','etoile_2']]

# flip rows from the oldest one to the newest
df_ligth = df_ligth[::-1].reset_index(drop=True)

# Add frequencies columns
df_extend = count_frequency(df_ligth, windows = windows)
# Add column with how long the number didn't not appear
df_extend = quadra_dif(df_extend)
# create columns counting number of draft without this number or star
df_extend = no_ball(df_extend)
df_extend = no_star(df_extend)

# Create sequences for X and y
X, y = create_sequences(df_extend, windows)

#data X for prediction
predictions = model.predict(X[-1:])

# Call Post traitement function
post_traitement(predictions)
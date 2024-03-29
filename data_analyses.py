# -*- coding: utf-8 -*-
"""Data analyses.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KUl0PJLpXhtsp4yBWkeBptcHchjbajyf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading dataset
df = pd.read_csv('euromillions_Fusion.csv',sep=';', encoding='ISO-8859-1')

df.head()

df.shape

column_names = df.columns
column_names

df.describe()

df.describe(include = ['object'])

df.info()

df.isna().sum()

df_ligth = df[['boule_1','boule_2','boule_3','boule_4','boule_5','etoile_1','etoile_2']]

test = df[['boules_gagnantes_en_ordre_croissant','etoiles_gagnantes_en_ordre_croissant']]

test.head(2)

df_ligth.head()

for column in df_ligth.columns:
    counts = df_ligth[column].value_counts()
#    print(f"\nDistribution of {column}:\n{counts}")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

bin_edges = [edge - 0.5 for edge in range(df_ligth.min().min(), df_ligth.max().max() + 2)]
# Histogram for features 1 to 5
axes[0].hist(df_ligth.iloc[:, :5].values.flatten(), bins=bin_edges, color='skyblue', edgecolor='black', width=0.95)
axes[0].set_title('Distribution of Numbers (Features 1 to 5)')
axes[0].set_xlabel('Number')
axes[0].set_ylabel('Frequency')
axes[0].set_xlim([0, 51])

# Histogram for features 6 and 7
hist_2 = axes[1].hist(df_ligth.iloc[:, 5:].values.flatten(), bins=bin_edges, color='skyblue', edgecolor='black', width=0.8)
axes[1].set_title('Distribution of stars')
axes[1].set_xlabel('Number')
axes[1].set_ylabel('Frequency')

# Customize x-axis labels for the second subplot
# axes[1].set_xticks(hist_2[1])
# axes[1].set_xticklabels(hist_2[1])
axes[1].set_xlim([0, 13])

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()

# Select features 1 to 5
features_1_to_5 = df_ligth.iloc[:, :5]
# Check if at least one value in each row is less than 10
at_least_one_less_than_10 = (features_1_to_5 < 10).any(axis=1)

# Count the number of rows where at least one value is less than 10
count_rows_at_least_one_less_than_10 = at_least_one_less_than_10.sum()/df_ligth.shape[0]*100

print(f'% of times a value < 10 appears numbers: {count_rows_at_least_one_less_than_10:.2f}%')

# Check if at least one value in each row is less than 20
at_least_one_less_than_20 = ((features_1_to_5 >= 10) & (features_1_to_5 < 20)).any(axis=1)

# Count the number of rows where at least one value is less than 20
count_rows_at_least_one_less_than_20 = at_least_one_less_than_20.sum()/df_ligth.shape[0]*100

print(f'% of times a value < 20 appears numbers: {count_rows_at_least_one_less_than_20:.2f}%')

# Check if at least one value in each row is less than 30
at_least_one_less_than_30 = ((features_1_to_5 >= 20) & (features_1_to_5 < 30)).any(axis=1)

# Count the number of rows where at least one value is less than 30
count_rows_at_least_one_less_than_30 = at_least_one_less_than_30.sum()/df_ligth.shape[0]*100

print(f'% of times a value < 30 appears numbers: {count_rows_at_least_one_less_than_30:.2f}%')

# Check if at least one value in each row is less than 40
at_least_one_less_than_40 = ((features_1_to_5 >= 30) & (features_1_to_5 < 40)).any(axis=1)

# Count the number of rows where at least one value is less than 40
count_rows_at_least_one_less_than_40 = at_least_one_less_than_40.sum()/df_ligth.shape[0]*100

print(f'% of times a value < 40 appears numbers: {count_rows_at_least_one_less_than_40:.2f}%')

# Check if at least one value in each row is less than 50
at_least_one_less_than_50 = ((features_1_to_5 >= 40) & (features_1_to_5 <= 50)).any(axis=1)

# Count the number of rows where at least one value is less than 50
count_rows_at_least_one_less_than_40 = at_least_one_less_than_50.sum()/df_ligth.shape[0]*100

print(f'% of times a value <= 50 appears numbers: {count_rows_at_least_one_less_than_40:.2f}%')

# Check if at least one value in each row is less than 50
at_least_one_less_than_25 = ((features_1_to_5 <= 25)).any(axis=1)

# Count the number of rows where at least one value is less than 50
count_rows_at_least_one_less_than_25 = at_least_one_less_than_25.sum()/df_ligth.shape[0]*100

print(f'% of times a value <= 25 appears numbers: {count_rows_at_least_one_less_than_25:.2f}%')

Number50 = (features_1_to_5 == 50).any(axis=1)
Number50 = Number50.sum()/df_ligth.shape[0]*100
print(f'% 50: {Number50:.2f}%')

Number1 = (features_1_to_5 == 1).any(axis=1)
Number1 = Number1.sum()/df_ligth.shape[0]*100
print(f'% 1: {Number1:.2f}%')

# Count the number of values less than 24 for each row in features 1 to 5
count_less_than_25_per_row = (features_1_to_5 <= 25).sum(axis=1)

# Print the count for each row
print('Number of values less than 25 for each row:')
print(count_less_than_25_per_row)

# Plot histogram
plt.figure(figsize=(8, 5))
bin_edges = [edge - 0.4 for edge in range(count_less_than_25_per_row.min(), count_less_than_25_per_row.max() + 2)]
plt.hist(count_less_than_25_per_row, bins=bin_edges, color='skyblue', edgecolor='black', width=0.8)
plt.title('Distribution of Values Less Than 25 Per Row')
plt.xlabel('Count of Values Less Than 25')
plt.ylabel('Frequency')
plt.xticks(range(6))
plt.show()
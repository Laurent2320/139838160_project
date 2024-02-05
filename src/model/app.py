import pandas as pd

def read_data(file_path, encoding='utf-8'):
    try:
        df = pd.read_csv(file_path, delimiter=';', encoding=encoding)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def preprocess_data(df):
    # Filter columns starting with "boule_" and "etoile_"
    selected_columns = [col for col in df.columns if col.startswith(('boule_', 'etoile_'))]
    filtered_df = df[selected_columns]
    return filtered_df

def main():
    file_path = 'data/euromillions_202002.csv' 
    encoding = 'latin-1' 

    # Step 1: Read data
    data = read_data(file_path, encoding)
    if data is None:
        return

    # Step 2: Preprocess data
    processed_data = preprocess_data(data)

    # Display the processed data
    print("Processed Data:")
    print(processed_data.head())
    print(processed_data)

if __name__ == "__main__":
    main()




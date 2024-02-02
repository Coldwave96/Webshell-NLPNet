import re
import os
import chardet
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import *

def process_code(source_code):
    processed_code = source_code.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    processed_code = re.sub(r'\s+', ' ', processed_code)
    return processed_code

def process_source_code_file(file_path):
    with open(file_path, 'rb') as file:
        rawdata = file.read()
        result = chardet.detect(rawdata)
        encoding = result['encoding']

    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
            source_code = file.read()         

    processed_code = process_code(source_code)
    return processed_code

def embedding_folder(model, folder_path, file_extensions):
    embedding_df = pd.DataFrame()

    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                processed_code = process_source_code_file(file_path)
                embedding_code = model.encode(processed_code, convert_to_numpy=True)

                temp_df = pd.DataFrame({'file_path': file_path, 'embedding_code': [embedding_code]})
                embedding_df = pd.concat([embedding_df, temp_df], ignore_index=True)

    return embedding_df

def process_train_folder():
    embedding_model = SentenceTransformer(embedding_model_name, cache_folder=cache_folder)

    # black_df = embedding_folder(embedding_model, test_black_path, file_extensions)
    # white_df = embedding_folder(embedding_model, test_white_path, file_extensions)

    black_df = embedding_folder(embedding_model, target_black_path, file_extensions)
    white_df = embedding_folder(embedding_model, target_white_path, file_extensions)
    black_df['label'] = 1
    white_df['label'] = 0
    samples_df = pd.concat([black_df, white_df], ignore_index=True)

    if not os.path.exists(output_samples_csv_folder):
        os.mkdir(output_samples_csv_folder)
    output_samples_csv_file = output_samples_csv_folder + f'black{len(black_df)}_white{len(white_df)}.csv'
    samples_df.to_csv(output_samples_csv_file)
    return samples_df

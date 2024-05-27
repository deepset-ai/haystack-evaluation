import re
import os

import pandas as pd


def parse_results(f_name: str):
    pattern = r"score_report_(.*?)__top_k:(\d+)__chunk_size:(\d+)\.csv"
    match = re.search(pattern, f_name)
    if match:
        embeddings_model = match.group(1)
        top_k = int(match.group(2))
        chunk_size = int(match.group(3))
        return embeddings_model, top_k, chunk_size
    else:
        print("No match found")


def read_results(path: str):
    all_scores = []
    for root, dirs, files in os.walk(path):
        for f_name in files:
            if not f_name.startswith("score_report"):
                continue

            embeddings_model, top_k, chunk_size = parse_results(f_name)

            df = pd.read_csv(path+"/"+f_name)

            df.rename(columns={'Unnamed: 0': 'metric'}, inplace=True)
            df_transposed = df.T
            df_transposed.columns = df_transposed.iloc[0]
            df_transposed = df_transposed[1:]

            # Add new columns
            df_transposed['embeddings'] = embeddings_model
            df_transposed['top_k'] = top_k
            df_transposed['chunk_size'] = chunk_size

            all_scores.append(df_transposed)

    df = pd.concat(all_scores)
    df.reset_index(drop=True, inplace=True)
    df.rename_axis(None, axis=1, inplace=True)

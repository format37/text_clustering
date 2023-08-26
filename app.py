import numpy as np
import pandas as pd
# import polars as pl
# import tiktoken
# import openai
# from openai.embeddings_utils import get_embedding
from datetime import datetime
# from ast import literal_eval
# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# import matplotlib
# import matplotlib.pyplot as plt
import logging
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import json
# from sklearn.metrics import pairwise_distances_argmin_min
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Log numpy version
# logger.info('Numpy version: {}'.format(np.__version__))
# Log pandas version
# logger.info('Pandas version: {}'.format(pd.__version__))

logger.info('Loading model from HuggingFace Hub...')
# Define the device (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Loading model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# Move the model to the specified device
model.to(device)
logger.info('Model loaded.')


# Mean Pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_sentence_embeddings(df):
    # Sorting values
    # top_n = 1000
    df = df.sort_values("linkedid")
    # df = df.sort_values("linkedid").tail(top_n * 2)

    # Ensure that all entries in 'text' column are strings
    sentences = df['text'].astype(str).tolist()

    # Create DataLoader for batching
    batch_size = 32  # Adjust based on your GPU's memory
    dataloader = DataLoader(sentences, batch_size=batch_size, shuffle=False)

    embeddings_list = []
    
    for batch in dataloader:
        # Tokenize sentences
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

        # Move the encoded input to the specified device
        encoded_input = encoded_input.to(device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Apply mean pooling to get sentence embeddings
        attention_mask = encoded_input['attention_mask']
        embeddings = mean_pooling(model_output, attention_mask).cpu().numpy()
        embeddings_list.extend(embeddings)

    # Add embeddings to the DataFrame
    df['embedding'] = embeddings_list
    # df = df.tail(top_n)  # Keep only the top_n entries

    return df
    

def main():
    # openai_key = input('Enter OpenAI key: ')
    # dataset_path = '../../datasets/transcribations/transcribations_2023-04-27 16:07:39_2023-07-25 19:03:21_polars.csv'
    # n_clusters = 4
    # Load calls and format to conversations
    # df = calls_to_converations(dataset_path, '2023-07-21', n_from=1, n_to=5)
    # df.to_csv('conversations.csv')
    
    # Load conversations
    # df = pd.read_csv('conversations.csv')
    df = pd.read_csv('data/data.csv')
    
    # Ada v2	$0.0001 / 1K tokens
    
    # df = get_embeddings(df, openai_key=openai_key)
    df = get_sentence_embeddings(df)
    df.to_csv("data/embeddings.csv")
    # Load embeddings
    # df = pd.read_csv('embeddings.csv')
    # df = pd.read_csv('local_conversations_embeddings.csv')

    # Reloading the original DataFrame from the CSV file
    # df = pd.read_csv('embeddings.csv')
    # Applying the custom conversion function to the 'embedding' column
    # df['embedding'] = df['embedding'].apply(convert_to_array)
    # Clustering
    # df, matrix = clustering(df, n_clusters=n_clusters)

    # Summarize topics
    # legend = topic_samples_central(df, matrix, openai_key=openai_key, n_clusters=n_clusters, rev_per_cluster=10)
    # Fake legend
    # legend = ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 3']
    # Plot clusters
    # plot_clusters(df, matrix, legend_append_values=legend)

    logger.info('Done.')


if __name__ == "__main__":
    main()

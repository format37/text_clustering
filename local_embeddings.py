import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import pickle
import numpy as np


def get_text_embedding(text, embeddings_dict):
    words = text.split()
    word_embeddings = [embeddings_dict[word] for word in words if word in embeddings_dict]
    if not word_embeddings:
        return None  # Return None if the text has no words that are in the vocabulary
    text_embedding = np.mean(word_embeddings, axis=0)
    return text_embedding.tolist()  # convert numpy array to list


def main():

    # Define some constants
    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 100
    EPOCHS = 400
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001

    # Load the data
    data = pd.read_csv('conversations.csv')

    # Preprocess the data
    # We'll start by tokenizing the text
    data['tokenized'] = data['text'].apply(lambda x: x.split())

    # Build the vocabulary
    vocab = set(word for conversation in data['tokenized'] for word in conversation)
    vocab_size = len(vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    # Prepare the training data
    ngrams = []
    for conversation in data['tokenized']:
        ngrams += [
            (
                [conversation[i - j - 1] for j in range(CONTEXT_SIZE)],
                conversation[i]
            )
            for i in range(CONTEXT_SIZE, len(conversation))
        ]

    # Split the data into training and validation sets
    ngrams_train, ngrams_val = train_test_split(ngrams, test_size=0.2, random_state=42)

    # Define the model
    class NGramLanguageModeler(nn.Module):

        def __init__(self, vocab_size, embedding_dim, context_size):
            super(NGramLanguageModeler, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear1 = nn.Linear(context_size * embedding_dim, 128)
            self.linear2 = nn.Linear(128, vocab_size)

        def forward(self, inputs):
            # embeds = self.embeddings(inputs).view((1, -1))
            embeds = self.embeddings(inputs).view((-1, CONTEXT_SIZE * EMBEDDING_DIM))
            out = F.relu(self.linear1(embeds))
            out = self.linear2(out)
            log_probs = F.log_softmax(out, dim=1)
            return log_probs

    # Instantiate the model and define the loss function and optimizer
    model = NGramLanguageModeler(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
    model = model.cuda()  # Move model to GPU
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Function to prepare a batch of data
    def prepare_batch(ngrams_batch):
        contexts = []
        targets = []
        for context, target in ngrams_batch:
            contexts.append([word_to_ix[word] for word in context])
            targets.append(word_to_ix[target])
        return torch.tensor(contexts, dtype=torch.long).cuda(), torch.tensor(targets, dtype=torch.long).cuda()  # Move data to GPU

    # Prepare the data loaders
    train_loader = DataLoader(ngrams_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=prepare_batch)
    val_loader = DataLoader(ngrams_val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=prepare_batch)

    # Train the model
    for epoch in range(EPOCHS):
        model.train()
        total_loss_train = 0
        for contexts, targets in train_loader:
            model.zero_grad()
            log_probs = model(contexts)
            loss = loss_function(log_probs, targets)
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()

        model.eval()
        total_loss_val = 0
        with torch.no_grad():
            for contexts, targets in val_loader:
                log_probs = model(contexts)
                loss = loss_function(log_probs, targets)
                total_loss_val += loss.item()

        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Training loss: {total_loss_train / len(train_loader)}')
        print(f'Validation loss: {total_loss_val / len(val_loader)}')

    # Extract the embeddings
    embeddings = model.embeddings.weight.detach().cpu().numpy()
    embeddings_dict = {word: embeddings[i] for word, i in word_to_ix.items()}
    # Save the embeddings_dict
    with open('embeddings_dict.pkl', 'wb') as f:
        pickle.dump(embeddings_dict, f)
    print('Embeddings saved!')

    # Load the embeddings_dict
    # with open('embeddings_dict.pkl', 'rb') as f:
    #     embeddings_dict = pickle.load(f)

    # Print first 5 keys
    print(list(embeddings_dict.keys())[:5])
    
    print('Computing embeddings for the conversations...')
    # Add an 'embedding' column to the DataFrame
    data['embedding'] = data['text'].apply(get_text_embedding, args=(embeddings_dict,))

    print('Saving the DataFrame...')
    # Save the DataFrame to csv
    data.to_csv('local_conversations_embeddings.csv', index=False)
    print('Done!')


if __name__ == '__main__':
    main()

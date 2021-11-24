#TODO megre this file and "Main.py"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from nltk import tokenize
from Tweet_to_OF import count_all, remove_untokenizable, get_backup_tweets
from collections import Counter


def save_model(model):
    torch.save(model.state_dict(), "./model.pth")


def load_model():
    model = torch.load("./model.pth")
    model.eval()
    return model


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def substitute_with_unk(data, n=1):
    count = Counter()
    for word in data:
        count += Counter(word)

    for i in range(len(count)):
        if count[data[i]] <= n:
            data[i] = "<UNK>"
    return data


def get_twitter_data_tokenized(file="testtweets"):
    with open(".\\tweets\\"+ file, "r", encoding="utf-8") as f:
        return tokenize.word_tokenize(f.read())


def get_opin_data():
    return count_all()


class LSTMmodel(nn.Module):
    # Class that defines our model
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMmodel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    # This is the forward computation, which constructs the computation graph
    def forward(self, sentence):
        # Get the embeddings
        embeds = self.word_embeddings(sentence)
        # put them through the LSTM and get its output
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # pass that output through the linnear layer
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # convert the logits to a log probability distribution
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


get_backup_tweets("sourcetesttweets", "testtweets")
remove_untokenizable("./tweets/sourcetesttweets", "./tweets/testtweets")
train_data_unk = substitute_with_unk(get_twitter_data_tokenized())
print(train_data_unk)


word_to_ix = {}
ix_to_word = {}
tag_to_ix = {}
ix_to_tag = {}
for sent, tags in train_data_unk:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            ix_to_word[word_to_ix[word]] = word
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
            ix_to_tag[tag_to_ix[tag]] = tag

EMBEDDING_DIM = 32
HIDDEN_DIM = 32
# LAYERS =

# Initialize the model
model = LSTMmodel(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
# Loss function to use
loss_function = nn.NLLLoss()
# Optimizer to use during training
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(train_data_unk[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
    for i, word in enumerate(train_data_unk[0][0]):
        j = int(np.argmax(tag_scores[i]))
        print(f"\t{word}|{ix_to_tag[j]}")

# Training loop

for epoch in range(20):  # normally you would NOT do 100 epochs, it is toy data
    print(f"Starting epoch {epoch}...")
    loss_sum = 0
    correct = 0
    for sentence, tags in train_data_unk:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        # Eventually I suggest you use the DataLoader modules
        # The batching can take place here
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)
        # print(tag_scores)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss_sum += loss.data.item()
        for i in range(len(tag_scores)):
            val = int(np.argmax(tag_scores[i].detach()))
            if ix_to_tag[val] in tags:
                correct += 1
        loss.backward()
        optimizer.step()
    print("Epoch {}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch, loss_sum, correct / len(tag_scores)))

import warnings
import nltk
import torch.nn as nn
import torch

torch.manual_seed(420)

nltk.download('stopwords')
nltk.download('punkt')
warnings.filterwarnings("ignore")

class TextClassificationModel(nn.Module):
    def __init__(self, num_classes, num_words):
        super(TextClassificationModel, self).__init__()
        embed_dim = 1024
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=embed_dim)
        self.dropout = nn.Dropout(p=0.6)

        self.lstm = nn.LSTM(embed_dim, 128, bidirectional=True, batch_first=True, num_layers=2, dropout=0.5)
        # output layer is a layer which has only one output
        # input(512) = 128+128 for mean and same for max pooling
        self.out = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Softmax()
        )

    def forward(self, text):
        x = self.embedding(text)
        x = self.dropout(x)
        # move the embedding output to lstm
        x,_ = self.lstm(x)
        # apply mean and max pooling on lstm output
        avg_pool = torch.mean(x,1)
        max_pool, _ = torch.max(x,1)
        # concatenate mean and max pooling this is why 512
        # 128 for each direction = 256
        # avg_pool = 256, max_pool = 256
        out = torch.cat((avg_pool,max_pool), 1)
        # pass through the output layer and return the output
        out = self.out(out)
        return out
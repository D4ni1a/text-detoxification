import warnings
import nltk
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from predict_model import predict
from train_span_model import train_span_model

torch.manual_seed(420)

nltk.download('stopwords')
nltk.download('punkt')
warnings.filterwarnings("ignore")

directory_binary = "./data/interm"
filename_binary = directory_binary + "/toxic_binary.csv"

train_bin = pd.read_csv(filename_binary, index_col = 0)
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
labels = ["toxic", "non-toxic"]
hypothesis_template = 'This text is {}.'
sent_tranf_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

model = train_span_model()

final_sim = 0
final_tox = 0
initial_tox = 0
size = 1000
for i in tqdm(train_bin.index[:size]):
    if train_bin.iloc[i]['label']:
        sent = train_bin.iloc[i]['sentence']

        initial_tox += 1

        peref = predict(sent, model)
        classified = classifier(peref, labels,
                                hypothesis_template=hypothesis_template)

        final_tox += classified['labels'][np.argmax(classified['scores'])] == 'toxic'
        text_embeddings = sent_tranf_model.encode([sent, peref])
        final_sim += int(util.pytorch_cos_sim(text_embeddings[0], text_embeddings[1])[0][0])

print(f'Lower toxicity on {1 - final_tox/initial_tox}')
print(f'Initial toxicity: {initial_tox/size}\nFinal toxicity: {final_tox/size}\nFinal similarity: {final_sim/size}')
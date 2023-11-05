import warnings
import nltk
import torch
from transformers import pipeline
from toxic_parafraser import ToxicParafraser

torch.manual_seed(420)

nltk.download('stopwords')
nltk.download('punkt')
warnings.filterwarnings("ignore")

global vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(sentence, model):
    tx_par = ToxicParafraser(model,
                             pipeline('zero-shot-classification', model='facebook/bart-large-mnli'))
    return tx_par.forward(sentence)
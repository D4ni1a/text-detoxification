import warnings
import nltk
from nltk.tokenize import word_tokenize
import re
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, util
from IPython.display import clear_output
from train_tagger_model import test_span_model

torch.manual_seed(420)

nltk.download('stopwords')
nltk.download('punkt')
warnings.filterwarnings("ignore")

global vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ToxicParafraser():
    def __init__(self, span_model, classifier, model = "distilbert-base-uncased",
                 sent_model = 'distilbert-base-nli-mean-tokens'):
        super().__init__()
        self.labels_ = ["toxic", "non-toxic"]
        self.hypothesis_template_ = 'This text is {}.'
        self.classifier = classifier
        self.span_model = span_model
        self.nlp_model = AutoModelForMaskedLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.sent_tranf_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    def masker(self, sentence, mask, tag = "[MASK]"):
        if sum(mask) == 0:
            return sentence, ""
        else:
            sentence_result = {}
            full_sent = ""
            full_mask = ""
            tokenized = word_tokenize(sentence)
            i = 0
            for word in tokenized:
                result = re.match('^[\W]*$', word)
                if result is None:
                    if mask[i]:
                        for k in sentence_result.keys():
                            sentence_result[k] += " "*(i!=0) + word
                        sentence_result[i] = full_sent + " "*(i!=0) + tag
                        full_sent += " "*(i!=0) + word
                        full_mask += " "*(i!=0) + tag
                    else:
                        if not (re.match('^[\W]', word) is None):
                            add = word
                        else:
                            add = " "*(i!=0) + word

                        for k in sentence_result.keys():
                            sentence_result[k] += add

                        full_sent += add
                        full_mask += add
                    i += 1
                else:
                    for k in sentence_result.keys():
                        sentence_result[k] += word
                    full_sent += word
                    full_mask += word

            return sentence_result, full_mask

    def span_predict(self, text):
        predictions = test_span_model(self.span_model, text)
        return predictions

    def replacement(self, text, words, tag = "[MASK]"):
        text_word = [text]
        best_text = ""
        best_sim = 0

        for word in words:
            replaced_text = text.replace(tag, word)
            text_word.append(replaced_text)

        text_embeddings = self.sent_tranf_model.encode(text_word)

        for i in range(1, len(text_embeddings)):
            sim = util.pytorch_cos_sim(text_embeddings[0], text_embeddings[i])
            if sim > best_sim:
                best_sim = sim
                best_text = i - 1

        return words[best_text]

    def forward(self, text, tag = "[MASK]"):
        prediction = self.classifier(text,
                                     self.labels_,
                                     hypothesis_template=self.hypothesis_template_,
                                     multi_label=False)
        if prediction["labels"][np.argmax(prediction["scores"])] == "non-toxic":
            return text

        mask = self.span_predict(text)
        masked_text, full_mask = self.masker(text, mask[0])

        if full_mask == "":
            return text

        for k in masked_text.keys():

            inputs = self.tokenizer(masked_text[k], return_tensors="pt")
            token_logits = self.nlp_model(**inputs).logits
            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

            mask_token_logits = token_logits[0, mask_token_index, :]

            candidates = torch.sort(mask_token_logits, dim=1, descending=True).indices[0].tolist()

            cand = 0
            resulted_arr = []
            for i, token in enumerate(candidates):
                if re.match('^[\W]*$', self.tokenizer.decode([token])) is not None:
                    continue

                replaced_text = masked_text[k].replace(self.tokenizer.mask_token,
                                                       self.tokenizer.decode([token]))
                replaced_mask = self.span_predict(replaced_text)
                clear_output()

                if replaced_mask[0][k] != 1:
                    resulted_arr.append(self.tokenizer.decode([token]))
                if len(resulted_arr) == 5:
                    break


            best_word = self.replacement(masked_text[k], resulted_arr)
            full_mask = full_mask.replace(tag, best_word, 1)

        return full_mask
import pandas as pd
import warnings
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
import ast
import re
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

torch.manual_seed(420)

nltk.download('stopwords')
nltk.download('punkt')
warnings.filterwarnings("ignore")

global vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Processing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

    def tokenize_text(self, text: str) -> list[str]:
        return word_tokenize(text)

    def remove_stop_words(self, tokenized_text: list[str]) -> list[str]:
        return [w for w in tokenized_text if not w.lower() in self.stop_words]

    def stem_words(self, tokenized_text: list[str]) -> list[str]:
        return [self.ps.stem(w) for w in tokenized_text]

def yield_tokens(df):
    for _, sample in df.iterrows():
        yield sample.to_list()[0]

def to_test_df(sentence):
    return_df = pd.DataFrame([], columns=['sentence'])
    for i in range(len(sentence)):
        tokenized = word_tokenize(sentence[i].lower())
        tmp = []
        for word in tokenized:
            result = re.match('^[\W]*$', word)
            if result is None:
                tmp.append(word)
        return_df = return_df.append({"sentence": tmp},
                                            ignore_index = True)
    return return_df

def lower_text(text: str):
    return text.lower()

def remove_numbers(text: str):
    """
    Substitute all punctuations with space in case of
    "there is5dogs".

    If subs with '' -> "there isdogs"
    With ' ' -> there is dogs
    """
    text_nonum = re.sub(r'\d+', ' ', text)
    return text_nonum

def remove_punctuation(text: str):
    """
    Substitute all punctiations with space in case of
    "hello!nice to meet you"

    If subs with '' -> "hellonice to meet you"
    With ' ' -> "hello nice to meet you"
    """
    text_nopunct = re.sub(r'[^a-z|\s]+', ' ', text)
    return text_nopunct

def remove_multiple_spaces(text: str):
    text_no_doublespace = re.sub('\s+', ' ', text).strip()
    return text_no_doublespace

def preprocessing_stage(text, pr = Processing()):
    _lowered = lower_text(text)
    _without_numbers = remove_numbers(_lowered)
    _without_punct = remove_punctuation(_without_numbers)
    _single_spaced = remove_multiple_spaces(_without_punct)
    _tokenized = pr.tokenize_text(_single_spaced)
    return _tokenized

def clean_text_inplace(df):
    df['sentence'] = df['sentence'].apply(preprocessing_stage)
    return df

def preprocess(df):
    df.fillna(" ", inplace=True)
    _cleaned = clean_text_inplace(df)
    return _cleaned

def preprocess_raw_data(file_path):
    df_raw = pd.read_csv(file_path, sep='\t', index_col=0)
    tmp_df = df_raw.drop(['similarity', 'lenght_diff'], axis=1)

    tmp_df['ref_tox'] = tmp_df['ref_tox'].apply(lambda x: 1 if x > 0.5 else 0)
    tmp_df['trn_tox'] = tmp_df['trn_tox'].apply(lambda x: 1 if x > 0.5 else 0)
    binary_df = pd.DataFrame([], columns=["sentence", "label"])
    binary_df["sentence"] = tmp_df['reference']
    binary_df["label"] = tmp_df['ref_tox']
    binary_df = pd.concat([binary_df,
                           tmp_df[['translation', 'trn_tox']].rename(columns={"translation": "sentence",
                                                                              "trn_tox": "label"})],
                          axis=0, ignore_index=True)

    train_preprocessed = preprocess(binary_df)
    return train_preprocessed

def build_toxic_train_val_dataloaders(raw_data = "../../data/raw/filtered.tsv",
                                      proc_data = "", ratio = 0.2):
    if proc_data == "":
        train_bin_prep = pd.read_csv(proc_data, index_col=0)
    else:
        train_bin_prep = preprocess_raw_data(raw_data)

    train_classif, val_classif = train_test_split(
        train_bin_prep, stratify=train_bin_prep['label'], test_size=ratio,
        random_state = 420
    )

    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    vocabulary = build_vocab_from_iterator(yield_tokens(train_classif), specials=special_symbols)
    vocabulary.set_default_index(UNK_IDX)

    def collate_batch_tr_val(batch):
        label_list, text_list, offsets = [], [], [0]
        for _text, _label in batch:
            label_list.append(_label)
            text_list.append(torch.tensor(vocabulary(ast.literal_eval(_text)), dtype=torch.int64))

        text_list = pad_sequence(text_list, batch_first=True, padding_value=1)
        return torch.tensor(label_list, dtype=torch.long).to(device), torch.tensor(text_list, dtype=torch.long).to(
            device), torch.tensor(offsets).to(device)

    train_dataloader = DataLoader(
        train_classif.to_numpy(), batch_size=128, shuffle=True, collate_fn=collate_batch_tr_val
    )

    val_dataloader = DataLoader(
        val_classif.to_numpy(), batch_size=128, shuffle=False, collate_fn=collate_batch_tr_val
    )
    return train_dataloader, val_dataloader

def build_toxic_test_dataloader(sentence):
    test = to_test_df(sentence)

    def collate_batch_test(batch):
        text_list, offsets = [], [0]
        for _text in batch:
            print(_text[0])
            text_list.append(torch.tensor(vocabulary(_text[0]), dtype=torch.int64))

        text_list = pad_sequence(text_list, batch_first=True, padding_value=1)
        return torch.tensor(text_list, dtype=torch.long).to(device), torch.tensor(offsets).to(device)

    test_dataloader = DataLoader(
        test.to_numpy(), batch_size=1, shuffle=True, collate_fn=collate_batch_test
    )
    return test_dataloader
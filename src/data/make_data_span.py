from tqdm import tqdm
import pandas as pd
import warnings
import nltk
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

class ToxicSpanDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, vocab = None, max_size=100):
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.dataframe = dataframe
        self._preprocess()
        self.vocab = vocab or self._create_vocab()
        self._postprocess()

    def _preprocess(self):

        self.dataframe['tokens'] = self.dataframe['tokens'].apply(str.lower)
        self.sentences = [self._get_sentence(idx) for idx in list(self.dataframe['sentence_id'].unique())]
        if 'toxic?' in self.dataframe.columns:
            self.tags = [self._get_labels(idx) for idx in list(self.dataframe['sentence_id'].unique())]

    def _postprocess(self):
        for i, sentence in enumerate(self.sentences):
            self.sentences[i] = self.vocab(sentence)

    def _create_vocab(self):
        vocab = build_vocab_from_iterator(self.sentences,
                                          specials=self.special_symbols)
        vocab.set_default_index(self.UNK_IDX)
        return vocab

    def _get_sentence(self, index: int) -> list:
        sent = list(self.dataframe.loc[self.dataframe['sentence_id'] == index]['tokens'])
        return sent

    def _get_labels(self, index: int) -> list:
        tags = list(self.dataframe.loc[self.dataframe['sentence_id'] == index]['toxic?'])
        tags = [int(tag) for tag in tags]
        return tags

    def __getitem__(self, index) -> tuple[list, list]:
        if 'toxic?' in self.dataframe.columns:
            return (self.sentences[index], self.tags[index])
        else:
            return self.sentences[index]

    def __len__(self) -> int:
        return len(self.sentences)

def toxic_subseq(indexes):
    if len(indexes) == 0:
        return []
    start = indexes[0]
    array = []
    for i in range(len(indexes)-1):
        if indexes[i] + 1 == indexes[i + 1]:
            continue
        else:
            array.append((start, indexes[i] + 1))
            start = indexes[i + 1]
    array.append((start, indexes[i + 1] + 1))
    return array

def toxifier(words, label):
    return_df = pd.DataFrame([], columns=['tokens', 'toxic?'])
    tokenized = word_tokenize(words)
    for word in tokenized:
        result = re.match('^[\W]*$', word)
        if result is None:
            return_df = return_df.append({"tokens": word.lower(),
                                          "toxic?": label},
                                         ignore_index = True)
    return return_df

def toxic_identifier(indexes, sentence):
    return_df = pd.DataFrame([], columns=['tokens', 'toxic?'])
    if len(indexes) == 0:
        return return_df
    start, _ = indexes[0]
    flag = start == 0
    end = 0
    i = 0
    while i < len(indexes):
        if flag:
            start, end = indexes[i]
            end += 1
            i += 1
        else:
            if end == 0:
                new_end, _ = indexes[i]
                start, end = end, new_end
            else:
                new_end, _ = indexes[i]
                start, end = end - 1, new_end

        return_df = return_df.append(toxifier(sentence[start:end], flag),
                                     ignore_index=True)
        flag = not flag
    if end != len(sentence):
        return_df = return_df.append(toxifier(sentence[end:len(sentence)], flag),
                                     ignore_index=True)
    return return_df

def preprocess_raw_data(file_path):
    df_tsd_span = pd.read_csv(file_path)
    new_df = pd.DataFrame([], columns=['sentence_id', 'tokens', 'toxic?'])
    for i in tqdm(range(len(df_tsd_span))):
        toxic_indexes = ast.literal_eval(df_tsd_span['spans'].iloc[i])
        sentence = df_tsd_span['text'].iloc[i]
        toxic_subs = toxic_subseq(toxic_indexes)
        return_df = toxic_identifier(toxic_subs, sentence)
        return_df.insert(0, "sentence_id", i, True)
        new_df = new_df.append(return_df, ignore_index=True)
    return new_df

def to_test_df(sentence):
    return_df = pd.DataFrame([], columns=['sentence_id', 'tokens'])
    for i in range(len(sentence)):
        tokenized = word_tokenize(sentence[i])
        for word in tokenized:
            result = re.match('^[\W]*$', word)
            if result is None:
                return_df = return_df.append({"sentence_id": i,
                                            "tokens": word.lower()},
                                            ignore_index = True)
    return return_df

def build_span_train_val_dataloaders(raw_data="../../data/external/train_tsd_toxic_span.csv",
                                      proc_data="", ratio=0.2):
    if proc_data != "":
        train_span = pd.read_csv(proc_data, index_col=0)
    else:
        train_span = preprocess_raw_data(raw_data)

    train_split, val_split = train_test_split(range(train_span['sentence_id'].max()),
                                              test_size=ratio, random_state=420)

    train_dataframe = train_span[train_span['sentence_id'].isin(train_split)]
    val_dataframe = train_span[train_span['sentence_id'].isin(val_split)]

    # Create train dataset
    train_dataset = ToxicSpanDataset(train_dataframe)
    val_dataset = ToxicSpanDataset(val_dataframe, vocab=train_dataset.vocab)

    batch_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def collate_batch(batch: list):
        sentences_batch, postags_batch = [], []
        for _sent, _postags in batch:
            sentences_batch.append(torch.tensor(_sent, dtype=torch.int64))
            postags_batch.append(torch.tensor(_postags, dtype=torch.int64))

        sentences_batch = pad_sequence(sentences_batch, batch_first=True, padding_value=1).T
        postags_batch = pad_sequence(postags_batch, batch_first=True, padding_value=0).T
        postags_batch = torch.unsqueeze(postags_batch, 2)

        return torch.tensor(sentences_batch, dtype=torch.long).to(device), torch.tensor(postags_batch,
                                                                                        dtype=torch.long).to(device)
    global vocabulary
    vocabulary = train_dataset.vocab

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    return train_dataloader, val_dataloader, vocabulary


def build_span_test_dataloader(sentence):
    batch_size = 1

    def collate_batch(batch: list):
        sentences_batch, sentences_lengths = [], []
        for _sent in batch:
            sentences_batch.append(torch.tensor(_sent, dtype=torch.int64))
            sentences_lengths.append(len(_sent))

        sentences_batch = pad_sequence(sentences_batch, batch_first=True, padding_value=1).T
        return torch.tensor(sentences_batch, dtype=torch.long).to(device), sentences_lengths

    test = to_test_df(sentence)
    test_dataset = ToxicSpanDataset(test, vocab=vocabulary)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                                  collate_fn=collate_batch)

    return test_dataloader
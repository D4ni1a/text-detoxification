from tqdm import tqdm
import warnings
import nltk
import torch.nn as nn
import torch
import numpy as np
from toxic_tagger import ToxicTagger
from make_data_span import build_span_train_val_dataloaders

torch.manual_seed(420)

nltk.download('stopwords')
nltk.download('punkt')
warnings.filterwarnings("ignore")

global vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    epoch_num=-1
):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: train",
        leave=True,
    )
    model.train()
    train_loss = 0.0
    total = 0
    for i, batch in loop:
        texts, labels = batch

        optimizer.zero_grad()
        outputs = model(texts)
        output_dim = outputs.shape[-1]

        outputs = outputs[1:].view(-1, output_dim)
        labels = labels[1:].reshape(-1)
        loss = loss_fn(outputs, labels)

        # backward pass
        loss.backward()

        # optimizer run
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix({"loss": train_loss/(i * len(labels))})


def val_one_epoch(
    model,
    loader,
    loss_fn,
    epoch = 1,
    epoch_num=-1,
    best_so_far=0.0,
    ckpt_path='best.pt'
):

    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch}: val",
        leave=True,
    )
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()  # evaluation mode
        for i, batch in loop:
            texts, labels = batch

            outputs = model(texts)
            output_dim = outputs.shape[-1]

            outputs = outputs[1:].view(-1, output_dim)
            labels = labels[1:].reshape(-1)
            loss = loss_fn(outputs, labels)

            _, predicted = outputs.data.max(1, keepdim=True)
            total += labels.size(0)
            correct += predicted.eq(labels.data.view_as(predicted)).sum()

            val_loss += loss
            loop.set_postfix({"loss": val_loss/(total), "acc": correct / (total)})

        if correct / total > best_so_far:
            torch.save(model.state_dict(), ckpt_path)
            return correct / (total)

    return best_so_far

def predict(
    model,
    loader,
):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Predictions",
        leave=True,
    )
    predictions = []
    with torch.no_grad():
        model.eval()  # evaluation mode
        for i, batch in loop:
            texts, length = batch
            outputs = model(texts)

            _, predicted = torch.max(outputs.data, 2)
            predicted = predicted.T
            predicted = [np.array(torch.tensor(predicted[i,:length[i]], device='cpu')) for i, pred in enumerate(predicted.detach().cpu().tolist())]

            predictions += predicted

    return predictions

def train_span_model():
    train_dataloader, val_dataloader, vocab = build_span_train_val_dataloaders()
    INPUT_DIM = len(vocab)

    model = ToxicTagger(INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    best = -float('inf')
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, train_dataloader, optimizer, loss_fn,
                        epoch_num=epoch)
        best = val_one_epoch(model, val_dataloader, loss_fn, epoch,
                             best_so_far=best, ckpt_path="best_toxic_span.pt")
    return model

def test_span_model(model, sentences):
    test_dataloader = build_span_test_dataloader(sentences)
    predictions = predict(model, test_dataloader)
    return predictions
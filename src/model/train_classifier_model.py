from tqdm import tqdm
import warnings
import nltk
import torch.nn as nn
import torch
from text_classifier import TextClassificationModel
from make_data_bin import build_toxic_train_val_dataloaders, build_toxic_test_dataloader

torch.manual_seed(420)

nltk.download('stopwords')
nltk.download('punkt')
warnings.filterwarnings("ignore")

global vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch_classif(
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
    for i, batch in loop:
        labels, texts, offsets = batch
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(texts)
        loss = loss_fn(outputs, labels)

        # backward pass
        loss.backward()

        # optimizer run
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix({"loss": train_loss/(i * len(labels))})

def val_one_epoch_classif(
    model,
    loader,
    loss_fn,
    epoch_num=-1,
    best_so_far=0.0,
    ckpt_path='best.pt'
):

    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: val",
        leave=True,
    )
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()  # evaluation mode
        for i, batch in loop:
            labels, texts, offsets = batch

            # forward pass
            outputs = model(texts)
            # loss calculation
            loss = loss_fn(outputs, labels)

            _, predicted = outputs.data.max(1, keepdim=True)
            total += labels.size(0)
            correct += predicted.eq(labels.data.view_as(predicted)).sum()

            val_loss += loss
            loop.set_postfix({"loss": val_loss/total, "acc": correct / total})

        if correct / total > best_so_far:
            torch.save(model.state_dict(), ckpt_path)
            return correct / total

    return best_so_far

def predict(
    model,
    loader,
):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc="Predictions:",
        leave=True,
    )
    predictions = []
    with torch.no_grad():
        model.eval()  # evaluation mode
        for i, batch in loop:
            texts, offsets = batch

            # forward pass and loss calculation
            outputs = model(texts)

            _, predicted = torch.max(outputs.data, 1)
            predictions += predicted.detach().cpu().tolist()

    return predictions

def train_classifier_model():
    train_dataloader, val_dataloader, vocab = build_toxic_train_val_dataloaders()

    epochs = 3
    model = TextClassificationModel(2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best = -float('inf')
    for epoch in range(epochs):
        train_one_epoch_classif(model, train_dataloader, optimizer, loss_fn, epoch_num=epoch)
        best = val_one_epoch_classif(model, val_dataloader, loss_fn, epoch, best_so_far=best,
                                     ckpt_path="best_classifier.pt")
    return model

def test_classifier_model(model, sentences):
    test_dataloader = build_toxic_test_dataloader(sentences)
    predictions = predict(model, test_dataloader)
    return predictions
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torchvision.datasets
import torchvision.models

import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import accuracy_score
import datetime
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data_loaders(train_data: Dataset, test_data: Dataset, batch_size:int):
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_data_loader, test_data_loader

def test_model(model: nn.Module, data_loader: DataLoader, criterion):
    labels = []
    predictions = []
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in data_loader:
            image = data[0].to(device)
            label = data[1].to(device)
            pred = model(image)
            pred_class = torch.argmax(pred, dim=1)
            loss = criterion(pred, label)
            test_loss += loss.sum()

            for l, p in zip(label, pred_class):
                labels.append(l.item())
                predictions.append(p.item())
        return accuracy_score(labels, predictions), test_loss / len(data_loader.dataset)

def train_model(model: nn.Module, train_data_loader: DataLoader, test_data_loader: DataLoader, optimizer, criterion, model_path, num_epochs=50, model_weights=None):
    """
    Parameters:
    model - model to be trained \n
    data_loader - data loader containing the training data \n
    criterion - evaluation function to be used \n
    optimizer - optimizer to optimize the weights \n
    """
    if model_weights:
        model.load_state_dict(model_weights)
    model.to(device)

    train_loss = 0
    start_time = datetime.datetime.now()
    best_acc = 0
    train_record = []
    test_record = []
    scheduler = ReduceLROnPlateau(optimizer,patience=3, min_lr=1e-6)

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_data_loader)
        for data in loop:
            optimizer.zero_grad()
            image = data[0].to(device)
            label = data[1].to(device)
            pred = model(image)
            loss = criterion(pred, label)
            train_loss += loss

            loss.backward()

            loop.set_postfix({"loss": loss.item()})
            loop.set_description(f"Epoch {epoch+1}")

            optimizer.step()

        print(f"epoch: {epoch+1} done, loss: {train_loss / len(train_data_loader.dataset)}")
        acc, test_loss = test_model(model, test_data_loader, criterion)
        print(f"Acc: {acc}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{model_path}/model.bin")
        train_record.append(train_loss.item())
        test_record.append(test_loss.item())
        scheduler.step(train_loss)
        print(scheduler.get_last_lr())
        train_loss = 0
        

    end_time = datetime.datetime.now()
    print(f"Time taken: {(end_time - start_time).total_seconds()} seconds")
    return train_record, test_record

def train(model, model_path, train_data_loader, test_data_loader, num_epochs, optimizer='adam', lr=1e-3):
    assert optimizer in ['adam', 'sgd'], 'Not valid option'
    criterion = nn.CrossEntropyLoss()
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  #1e-4 to converge + Adam
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
    
    return train_model(model, train_data_loader, test_data_loader, optimizer, criterion, model_path, num_epochs=num_epochs)
    
def test(model, model_path, test_data_loader, criterion):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    return test_model(model, test_data_loader, criterion)

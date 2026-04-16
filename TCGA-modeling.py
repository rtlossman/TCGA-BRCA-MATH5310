"""
Deep Learning Models for Classification of RNA-seq Data:
MLP, CNN, and RNN Model Training and Evaluation

Rhys Lossman
Georgetown University
MATH 5310 Deep Learning
April 23, 2026
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_pca_data():
    train = pd.read_feather('tcga-brca-pca-train.feather')
    val = pd.read_feather('tcga-brca-pca-val.feather')
    test = pd.read_feather('tcga-brca-pca-test.feather')

    return train, val, test

def load_ae_data():
    train = pd.read_feather('tcga-brca-ae-train.feather')
    val = pd.read_feather('tcga-brca-ae-val.feather')
    test = pd.read_feather('tcga-brca-ae-test.feather')

    return train, val, test

def sample_MLP_space(n, seed=42):
    
    #discrete hyperparameter space
    dropout_list = [0,0.05,0.1,0.2]
    lr_list = [1e-3,5e-3,1e-4,5e-4]
    depth_list = [1,2,3,4]
    width_list = [128,256,512]

    rng = np.random.default_rng(seed)
    params = []
    
    for _ in range(n):
        dropout = rng.choice(dropout_list)
        lr = rng.choice(lr_list)
        depth = rng.choice(depth_list)
        width = rng.choice(width_list)

        if depth == 1:
            hidden_dims = [width]
        elif depth ==2:
            hidden_dims = [width, width//8]
        elif depth ==3:
            hidden_dims = [width, width//4, width//16]
        else:
            hidden_dims = [width, width//2, width//8, width//16]
    
        params.append({
            "dropout": dropout,
            "lr": lr,
            "depth": depth,
            "width": width,
            "hidden_dims": hidden_dims
        })
        
    return params

def make_DataLoaders(train_df, val_df, test_df, batch_size = 32):

    le = LabelEncoder()
    
    X_train = train_df.drop(columns="BRCA_Subtype_PAM50").values
    y_train = le.fit_transform(train_df["BRCA_Subtype_PAM50"])

    X_val = val_df.drop(columns="BRCA_Subtype_PAM50").values
    y_val = le.fit_transform(val_df["BRCA_Subtype_PAM50"])

    X_test = test_df.drop(columns="BRCA_Subtype_PAM50").values
    y_test = le.fit_transform(test_df["BRCA_Subtype_PAM50"])

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype =torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False
    )

    return train_loader, val_loader, test_loader, le.classes_



class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, num_classes):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(dims[-1], num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_MLP_model(model, optimizer, criterion, train_loader, val_loader, max_epochs=100, patience = 15):

    training_loss = []
    val_accuracy = []

    best_val_acc = 0
    epochs_no_improve = 0

    for i in range(max_epochs):
        model.train()
        avg_loss=0
        total = 0
        for x, y in train_loader:

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            avg_loss+= loss.item()
            total += y.size(0)

        
        training_loss.append(avg_loss/total)
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:

                logits = model(x)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)
            
        val_acc = correct/total
        val_accuracy.append(val_acc)
        
        if val_acc < best_val_acc:
            epochs_no_improve += 1
        else:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        
        if epochs_no_improve >= patience:
            print(f'early stopping reached after {i+1} epochs')
            print(f'best validation accuracy: {best_val_acc}')
            break

    plt.figure()
    plt.plot(training_loss, label = 'train loss')
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.title('training loss over time')
    plt.show()

    plt.figure()
    plt.plot(val_accuracy, label = 'validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('val accuracy over time')
    plt.show()

    model.load_state_dict(best_model_state)

    return model, best_val_acc, i+1


def train_MLP(train_pca_loader, val_pca_loader, train_ae_loader, val_ae_loader, params, batch_size = 32, seed = 42):
    
    results = []

    for i, param in enumerate(params):
        print(f'Training model {i+1}...')

        model_pca = FNN(
            input_dim = 250,
            hidden_dims = param['hidden_dims'],
            dropout = param['dropout'],
            num_classes = 5
        )

        model_ae = FNN(
            input_dim = 250,
            hidden_dims = param['hidden_dims'],
            dropout = param['dropout'],
            num_classes = 5
        )

        optimizer_pca = optim.Adam(model_pca.parameters(), lr = param['lr'])
        optimizer_ae = optim.Adam(model_ae.parameters(), lr = param['lr'])
        criterion = nn.CrossEntropyLoss()

        model_pca, best_val_acc_pca, epochs_pca = train_MLP_model(
            model_pca, optimizer_pca, criterion,
            train_pca_loader, val_pca_loader
        )

        model_ae, best_val_acc_ae, epochs_ae= train_MLP_model(
            model_ae, optimizer_ae, criterion,
            train_ae_loader, val_ae_loader
        )

        torch.save(model_pca.state_dict(), f'tcga-mlp_pca_{i}_acc_{best_val_acc_pca}.pt')
        torch.save(model_ae.state_dict(), f'tcga-mlp_ae_{i}_acc_{best_val_acc_ae}.pt')

        num_params = sum(p.numel() for p in model_pca.parameters())

        results.append({
            "dropout": param["dropout"],
            "lr": param["lr"],
            "depth": param["depth"],
            "hidden_dims": param["hidden_dims"],
            "pca_val_acc": best_val_acc_pca,
            "epochs_pca": epochs_pca,
            "ae_val_acc": best_val_acc_ae,
            "epochs_ae": epochs_ae,
            "num_params": num_params
        })

    return pd.DataFrame(results)

def metrics(results):
    print(f'average number of epochs for pca: {results['epochs_pca'].mean()}')
    print(f'average number of epochs for ae: {results['epochs_ae'].mean()}')
    plot_df = pd.DataFrame({
        "epochs": np.concatenate([results["epochs_pca"], results["epochs_ae"]]),
        "accuracy": np.concatenate([results["pca_val_acc"], results["ae_val_acc"]]),
        "embedding": ["PCA"]*len(results) + ["Autoencoder"]*len(results),
        "num_params": np.concatenate([results["num_params"], results["num_params"]])
    })

    plt.figure(figsize=(7,5))

    sns.scatterplot(
        data=plot_df,
        x="epochs",
        y="accuracy",
        hue="embedding",
        s=80
    )

    plt.xlabel("Epochs Trained (Early Stopping)")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Model Accuracy vs Training Duration")
    plt.show()

    sns.scatterplot(
        data=plot_df,
        x="num_params",
        y="accuracy",
        hue="embedding",
        s=80
    )

    plt.xscale('log')
    plt.xlabel("Number of Model Parameters")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Model Accuracy vs Parameter Count")
    plt.show()

def test_MLP(test_loader, model, classes):
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:

            logits = model(x)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(y)

    preds = torch.cat(all_preds).numpy()
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    report = classification_report(labels, preds)
    print(report)

    macro_auroc = roc_auc_score(labels, probs, average='macro', multi_class='ovr')
    print(f'macro auroc: {macro_auroc}')

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm,display_labels = classes)
    disp.plot()
    plt.show()

    
def sample_CNN_space(n, seed = 42):
    
    #discrete hyperparameter space
    dropout_list = [0,0.05,0.1,0.2]
    lr_list = [1e-3,5e-3,1e-4,5e-4]
    filter_list = [3,5]
    depth_list = [1,2,3]

    rng = np.random.default_rng(seed)
    params = []
    
    for _ in range(n):
        dropout = rng.choice(dropout_list)
        lr = rng.choice(lr_list)
        depth = rng.choice(depth_list)
        filter = rng.choice(filter_list)
    
        params.append({
            "dropout": dropout,
            "lr": lr,
            "depth": depth,
            "filter": filter
        })
        
    return params

def reshape_data(X):
    pad_width = 256 - X.shape[1]
    X = nn.functional.pad(X, (0,pad_width))
    return X.view(X.shape[0],1,16,16)

class CNN(nn.Module):
    def __init__(self, input_dim, depth, dropout, filter, num_classes):
        super().__init__()

        layers = []
        input_channel = 1

        for _ in range(depth):
            layers.append(nn.Conv2d(input_channel,input_channel*4,filter, padding = 'same'))
            layers.append(nn.MaxPool2d(2,2))
            input_channel*=4

        layers.append(nn.AdaptiveAvgPool2d((8,8)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_channel*8*8, 16))
        layers.append(nn.ReLU())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(16,num_classes))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_CNN_individual(model, optimizer, criterion, train_loader, val_loader, max_epochs=100, patience = 15):
    training_loss = []
    val_accuracy = []

    best_val_acc = 0
    epochs_no_improve = 0

    for i in range(max_epochs):
        model.train()
        avg_loss=0
        total = 0
        for x, y in train_loader:

            z = reshape_data(x)

            optimizer.zero_grad()
            logits = model(z)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            avg_loss+= loss.item()
            total += y.size(0)

        
        training_loss.append(avg_loss/total)
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:

                z = reshape_data(x)
                logits = model(z)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)
            
        val_acc = correct/total
        val_accuracy.append(val_acc)
        
        if val_acc < best_val_acc:
            epochs_no_improve += 1
        else:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        
        if epochs_no_improve >= patience:
            print(f'early stopping reached after {i+1} epochs')
            print(f'best validation accuracy: {best_val_acc}')
            break

    plt.figure()
    plt.plot(training_loss, label = 'train loss')
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.title('training loss over time')
    plt.show()

    plt.figure()
    plt.plot(val_accuracy, label = 'validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('val accuracy over time')
    plt.show()

    model.load_state_dict(best_model_state)

    return model, best_val_acc, i+1

def train_CNN(train_pca_loader, val_pca_loader, train_ae_loader, val_ae_loader, params, seed = 42):
    results = []

    for i, param in enumerate(params):
        print(f'Training model {i+1}...')

        model_pca = CNN(
            input_dim = 250, 
            depth = param['depth'],
            dropout = param['dropout'],
            filter = param['filter'],
            num_classes = 5
        )

        model_ae = CNN(
            input_dim = 250,
            depth = param['depth'],
            dropout = param['dropout'],
            filter = param['filter'],
            num_classes = 5
        )

        optimizer_pca = optim.Adam(model_pca.parameters(), lr = param['lr'])
        optimizer_ae = optim.Adam(model_ae.parameters(), lr = param['lr'])
        criterion = nn.CrossEntropyLoss()

        model_pca, best_val_acc_pca, epochs_pca = train_CNN_individual(
            model_pca, optimizer_pca, criterion,
            train_pca_loader, val_pca_loader
        )

        model_ae, best_val_acc_ae, epochs_ae= train_CNN_individual(
            model_ae, optimizer_ae, criterion,
            train_ae_loader, val_ae_loader
        )

        torch.save(model_pca.state_dict(), f'tcga-cnn_pca_{i}_acc_{best_val_acc_pca}.pt')
        torch.save(model_ae.state_dict(), f'tcga-cnn_ae_{i}_acc_{best_val_acc_ae}.pt')

        num_params = sum(p.numel() for p in model_pca.parameters())

        results.append({
            "dropout": param["dropout"],
            "lr": param["lr"],
            "depth": param["depth"],
            "filter": param['filter'],
            "pca_val_acc": best_val_acc_pca,
            "epochs_pca": epochs_pca,
            "ae_val_acc": best_val_acc_ae,
            "epochs_ae": epochs_ae,
            "num_params": num_params
        })

    return pd.DataFrame(results)

def test_CNN(test_loader, model, classes):
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:

            z = reshape_data(x)
            logits = model(z)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(y)

    preds = torch.cat(all_preds).numpy()
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    report = classification_report(labels, preds)
    print(report)

    macro_auroc = roc_auc_score(labels, probs, average='macro', multi_class='ovr')
    print(f'macro auroc: {macro_auroc}')

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm,display_labels = classes)
    disp.plot()
    plt.show()

def sample_RNN_space(n, seed = 42):
     #discrete hyperparameter space
    dropout_list = [0,0.05,0.1,0.2]
    lr_list = [1e-3,5e-3,1e-4,5e-4]
    type_list = ['RNN','GRU']
    depth_list = [1,2]
    bidirectional = [True, False]
    width_list = [32,64,128]

    rng = np.random.default_rng(seed)
    params = []
    
    for _ in range(n):
        dropout = rng.choice(dropout_list)
        lr = rng.choice(lr_list)
        depth = rng.choice(depth_list)
        type = rng.choice(type_list)
        direction = rng.choice(bidirectional)
        width = rng.choice(width_list)
    
        params.append({
            "dropout": float(dropout),
            "lr": float(lr),
            "depth": int(depth),
            "type": type,
            "direction": bool(direction),
            "width": int(width)
        })
        
    return params

class RNN(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_layers, dropout, bidirectional, type, num_classes):
        super().__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if type == "RNN":
            self.rnn = nn.RNN(
                input_size = input_dim,
                hidden_size= hidden_dim,
                num_layers= num_layers,
                batch_first= True,
                dropout = dropout if num_layers > 1 else 0,
                bidirectional = bidirectional
            )

        if type == "GRU":
            self.rnn = nn.GRU(
                input_size = input_dim,
                hidden_size = hidden_dim, 
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout if num_layers > 1 else 0,
                bidirectional = bidirectional
            )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*self.num_directions, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        output, h_n = self.rnn(x)

        if self.bidirectional:
            h_n = h_n.view(self.rnn.num_layers, 2, x.size(0), self.rnn.hidden_size)
            last_hidden = torch.cat((h_n[-1,0], h_n[-1,1]), dim=1)
        else:
            last_hidden = h_n[-1]

        return self.fc(last_hidden)

def train_RNN_individual(model, optimizer, criterion, train_loader, val_loader, max_epochs=100, patience = 15):
    training_loss = []
    val_loss = []
    val_accuracy = []

    best_val_acc = 0
    epochs_no_improve = 0

    for i in range(max_epochs):
        model.train()
        avg_loss=0
        avg_vloss=0
        total = 0
        for x, y in train_loader:

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            avg_loss+= loss.item()*y.size(0)
            total += y.size(0)

        
        training_loss.append(avg_loss/total)
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:

                logits = model(x)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

                vloss = criterion(logits, y)
                avg_vloss+= vloss.item()*y.size(0)
            
            val_loss.append(avg_vloss/total)
            
        val_acc = correct/total
        val_accuracy.append(val_acc)
        
        if val_acc <= best_val_acc:
            epochs_no_improve += 1
        else:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        
        if epochs_no_improve >= patience:
            print(f'early stopping reached after {i+1} epochs')
            print(f'best validation accuracy: {best_val_acc}')
            break

    plt.figure()
    plt.plot(training_loss, label = 'train loss')
    plt.plot(val_loss, label = "validation loss")
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.title('training loss over time')
    plt.show()

    plt.figure()
    plt.plot(val_accuracy, label = 'validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('val accuracy over time')
    plt.show()

    model.load_state_dict(best_model_state)

    return model, best_val_acc, i+1

def train_RNN(train_pca_loader, val_pca_loader, train_ae_loader, val_ae_loader, params, seed = 42):
    results = []

    for i, param in enumerate(params):
        print(f'Training model {i+1}...')

        model_pca = RNN(
            input_dim = 1,
            hidden_dim= param['width'],
            num_layers= param['depth'],
            dropout = param['dropout'] if param['depth'] > 1 else 0,
            bidirectional = param['direction'],
            type = param['type'],
            num_classes = 5
        )

        model_ae = RNN(
            input_dim = 1,
            hidden_dim= param['width'],
            num_layers= param['depth'],
            dropout = param['dropout'] if param['depth'] > 1 else 0,
            bidirectional = param['direction'],
            type = param['type'], 
            num_classes=5
        )

        optimizer_pca = optim.Adam(model_pca.parameters(), lr = param['lr'])
        optimizer_ae = optim.Adam(model_ae.parameters(), lr = param['lr'])
        criterion = nn.CrossEntropyLoss()

        model_pca, best_val_acc_pca, epochs_pca = train_RNN_individual(
            model_pca, optimizer_pca, criterion,
            train_pca_loader, val_pca_loader
        )

        model_ae, best_val_acc_ae, epochs_ae= train_RNN_individual(
            model_ae, optimizer_ae, criterion,
            train_ae_loader, val_ae_loader
        )

        torch.save(model_pca.state_dict(), f'tcga-rnn_pca_{i}_acc_{best_val_acc_pca}.pt')
        torch.save(model_ae.state_dict(), f'tcga-rnn_ae_{i}_acc_{best_val_acc_ae}.pt')

        num_params = sum(p.numel() for p in model_pca.parameters())

        results.append({
            "dropout": param["dropout"],
            "lr": param["lr"],
            "depth": param["depth"],
            "hidden_dim": param['width'],
            "pca_val_acc": best_val_acc_pca,
            "epochs_pca": epochs_pca,
            "ae_val_acc": best_val_acc_ae,
            "epochs_ae": epochs_ae,
            "num_params": num_params
        })

    return pd.DataFrame(results)

def test_RNN(test_loader, model, classes):
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:

            logits = model(x)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(y)

    preds = torch.cat(all_preds).numpy()
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    report = classification_report(labels, preds)
    print(report)

    macro_auroc = roc_auc_score(labels, probs, average='macro', multi_class='ovr')
    print(f'macro auroc: {macro_auroc}')

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm,display_labels = classes)
    disp.plot()
    plt.show()

def main():
    set_seed()

    train_pca, val_pca, test_pca = load_pca_data()
    train_ae, val_ae, test_ae = load_ae_data()

    train_pca_loader, val_pca_loader, test_pca_loader, classes = make_DataLoaders(train_pca, val_pca, test_pca)
    train_ae_loader, val_ae_loader, test_ae_loader, _ = make_DataLoaders(train_ae, val_ae, test_ae)

    FNN_params = sample_MLP_space(20)


    FNN_results = train_MLP(
        train_pca_loader,
        val_pca_loader,
        train_ae_loader,
        val_ae_loader,
        FNN_params
    )

    FNN_results.to_csv("mlp_architecture_search.csv", index=False)

    model1 = FNN(250, hidden_dims=[256,64,16], dropout = 0,num_classes=5)
    model1.load_state_dict(torch.load('tcga-mlp_ae_0_acc_0.8611111111111112.pt'))

    model14 = FNN(250, hidden_dims=[128,16],dropout=0.05,num_classes=5)
    model14.load_state_dict(torch.load('tcga-mlp_pca_13_acc_0.9351851851851852.pt'))

    test_MLP(test_pca_loader, model14, classes)
    test_MLP(test_ae_loader, model1, classes)

    # CNN_params = sample_CNN_space(20)

    CNN_results = train_CNN(
        train_pca_loader,
        val_pca_loader,
        train_ae_loader,
        val_ae_loader,
        CNN_params
    )

    CNN_results.to_csv("cnn_architecture_search.csv", index=False)

    metrics(CNN_results)

    model1cnn = CNN(250, depth=1, dropout=0.05, filter=5, num_classes=5)
    model1cnn.load_state_dict(torch.load('tcga-cnn_ae_1_acc_0.8611111111111112.pt'))

    model11cnn = CNN(250, depth=2, dropout=0, filter=5, num_classes=5)
    model11cnn.load_state_dict(torch.load('tcga-cnn_pca_11_acc_0.8703703703703703.pt'))

    test_CNN(test_pca_loader, model11cnn, classes)
    test_CNN(test_ae_loader, model1cnn, classes)

    RNN_params = sample_RNN_space(20)

    RNN_results = train_RNN(
        train_pca_loader,
        val_pca_loader,
        train_ae_loader,
        val_ae_loader,
        RNN_params
    )

    RNN_results.to_csv("rnn_architecture_search.csv", index=False)

    metrics(RNN_results)

    model9rnn = RNN(input_dim = 1,
                hidden_dim= 64,
                num_layers= 2,
                dropout = 0.1,
                bidirectional = True,
                type = 'RNN',
                num_classes = 5)
    model9rnn.load_state_dict(torch.load('tcga-rnn_pca_8_acc_0.8333333333333334.pt'))

    model18rnn = RNN(input_dim = 1,
                hidden_dim= 64,
                num_layers= 2,
                dropout = 0.2,
                bidirectional = False,
                type = 'GRU',
                num_classes = 5)
    model18rnn.load_state_dict(torch.load('tcga-rnn_ae_17_acc_0.8240740740740741.pt'))

    test_RNN(test_pca_loader, model9rnn, classes)
    test_RNN(test_ae_loader, model18rnn, classes)

main()
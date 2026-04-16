"""
Deep Learning Models for Classification of RNA-seq Data:
TCGA Preprocessing and Compression

Rhys Lossman
Georgetown University
MATH 5310 Deep Learning
April 23, 2026
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

def main():
    pca_train_name = 'tcga-brca-pca-train.feather'
    pca_path = Path(pca_train_name)
    ae_train_name = 'tcga-brca-ae-train.feather'
    ae_path = Path(ae_train_name)

    raw_name  = 'tcga-brca-expression-labeled.feather'
    raw_path = Path(raw_name)

    if pca_path.is_file() and ae_path.is_file():
        print('Data already preprocessed. Load and model in modeling script "tcga-modeling.py"')
    elif raw_path.is_file():
        print('Loading raw data and preprocessing.')

        data, labels = load_raw_data()
        training, validation, testing = train_val_test_split(data, labels)

        training_data = training.drop(columns ='BRCA_Subtype_PAM50')
        training_labels = training['BRCA_Subtype_PAM50']
        validation_data = validation.drop(columns ='BRCA_Subtype_PAM50')
        validation_labels = validation['BRCA_Subtype_PAM50']
        testing_data = testing.drop(columns ='BRCA_Subtype_PAM50')
        testing_labels = testing['BRCA_Subtype_PAM50']

        train_standard, val_standard, test_standard = preprocess(training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels)

        train_data = train_standard.drop(columns ='BRCA_Subtype_PAM50')
        train_labels = train_standard['BRCA_Subtype_PAM50']
        val_data = val_standard.drop(columns ='BRCA_Subtype_PAM50')
        val_labels = val_standard['BRCA_Subtype_PAM50']
        test_data = test_standard.drop(columns ='BRCA_Subtype_PAM50')
        test_labels = test_standard['BRCA_Subtype_PAM50']

        #PCA
        k = 250
        _, _, _ = PCA_reduction(train_data, train_labels, val_data, val_labels, test_data, test_labels, k)

        #Autoencoder
        model = train_AE()
        _, _, _ = write_AE_data()

        print('PCA and autoencoder reduced data written to current directory.')
    else:
        print('Could not locate raw or processed data files.')

def load_raw_data():
    data = pd.read_feather('tcga-brca-expression-labeled.feather')
    labels = data.loc[:, 'BRCA_Subtype_PAM50']
    data = data.drop(columns=['patient', 'BRCA_Subtype_PAM50'])
    return data, labels

def train_val_test_split(data, labels, train = 0.7, val = 0.1, test = 0.2, seed=42):
    data_train, rdata, label_train, rlabel = train_test_split(data, labels, train_size = train, random_state=seed, stratify=labels)
    data_val, data_test, label_val, label_test = train_test_split(rdata, rlabel, train_size = val/(val+test), random_state=seed, stratify = rlabel)
    
    training = pd.concat([label_train, data_train], axis=1)
    validation = pd.concat([label_val, data_val], axis = 1)
    test = pd.concat([label_test, data_test], axis= 1)

    training.to_feather('tcga-brca-training.feather')
    validation.to_feather('tcga-brca-validation.feather')
    test.to_feather('tcga-brca-test.feather')
    
    return training, validation, test

def load_train_val_test():
    train = pd.read_feather('tcga-brca-training.feather')
    val = pd.read_feather('tcga-brca-validation.feather')
    test = pd.read_feather('tcga-brca-test.feather')

    data_train = train.drop(columns='BRCA_Subtype_PAM50')
    label_train = train.loc[:,'BRCA_Subtype_PAM50']
    data_val = val.drop(columns='BRCA_Subtype_PAM50')
    label_val = val.loc[:,'BRCA_Subtype_PAM50']
    data_test = test.drop(columns='BRCA_Subtype_PAM50')
    label_test= test.loc[:,'BRCA_Subtype_PAM50']

    return data_train, label_train, data_val, label_val, data_test, label_test


def preprocess(train, label_train, val, label_val, test, label_test):
    mask = (train > 1).sum(axis=0) > 0.2*train.shape[0]
    filtered_train = train.loc[:, mask]
    filtered_val = val.loc[:, mask]
    filtered_test = test.loc[:, mask]

    train_sums = np.sum(filtered_train, axis=1)
    train_sums = np.maximum(train_sums, 1)
    train_CPM = filtered_train.div(train_sums, axis=0) * 1e6
    val_sums = np.sum(filtered_val, axis=1)
    val_sums = np.maximum(val_sums, 1)
    val_CPM = filtered_val.div(val_sums, axis=0) * 1e6
    test_sums = np.sum(filtered_test, axis=1)
    test_sums = np.maximum(test_sums, 1)
    test_CPM = filtered_test.div(test_sums, axis=0) * 1e6

    log_train = np.log2(train_CPM+1)
    log_val = np.log2(val_CPM+1)
    log_test = np.log2(test_CPM+1)

    scaler = StandardScaler()
    train_standard = pd.DataFrame(scaler.fit_transform(log_train), columns = filtered_train.columns,
        index = filtered_train.index)
    val_standard = pd.DataFrame(scaler.transform(log_val), columns = filtered_val.columns,
        index = filtered_val.index)
    test_standard = pd.DataFrame(scaler.transform(log_test), columns = filtered_test.columns,
        index= filtered_test.index)

    train_standardl = pd.concat([label_train, train_standard], axis=1)
    val_standardl = pd.concat([label_val, val_standard], axis=1)
    test_standardl = pd.concat([label_test, test_standard], axis=1)

    train_standardl.to_feather('tcga-brca-train-standardized.feather')
    val_standardl.to_feather('tcga-brca-val-standardized.feather')
    test_standardl.to_feather('tcga-brca-test-standardized.feather')

    return train_standardl, val_standardl, test_standardl

def load_standardized_data():
    train = pd.read_feather('tcga-brca-train-standardized.feather')
    val = pd.read_feather('tcga-brca-val-standardized.feather')
    test = pd.read_feather('tcga-brca-test-standardized.feather')
    return train, val, test

def PCA_analysis(data):
    pca = PCA()
    pca.fit(data)
    var = np.cumsum(pca.explained_variance_ratio_)
    
    n80 = np.argmax(var >= 0.8)
    n90 = np.argmax(var >= 0.9)
    n95 = np.argmax(var >= 0.95)
    
    print(f'80% variance capture: {n80}\n 90% variance capture: {n90}\n 95% variance capture: {n95}')
    
    n_dim = [100,250,500,750]
    for n in n_dim:
        print(f'variance captured with {n} components: {var[n-1]}')
    

def PCA_reduction(train, train_labels, val, val_labels, test, test_labels, k):
    print(train.shape)
    print(train_labels.shape)
    
    pca = PCA(n_components=k)
    PCA_train = pd.DataFrame(pca.fit_transform(train))
    PCA_val = pd.DataFrame(pca.transform(val))
    PCA_test = pd.DataFrame(pca.transform(test))

    print(PCA_train.shape)

    ptrain = pd.concat([train_labels.reset_index(drop=True), PCA_train.reset_index(drop=True)], axis=1)
    pval = pd.concat([val_labels.reset_index(drop=True), PCA_val.reset_index(drop=True)], axis =1)
    ptest = pd.concat([test_labels.reset_index(drop=True), PCA_test.reset_index(drop=True)], axis =1)

    print(ptrain.shape)


    ptrain.to_feather('tcga-brca-pca-train.feather')
    pval.to_feather('tcga-brca-pca-val.feather')
    ptest.to_feather('tcga-brca-pca-test.feather')
    
    return ptrain, pval, ptest

def load_PCA_data():
    train = pd.read_feather('tcga-brca-pca-train.feather')
    val = pd.read_feather('tcga-brca-pca-val.feather')
    test = pd.read_feather('tcga-brca-pca-test.feather')

    return train, val, test


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(33443,1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 250)
        )
    
        self.decoder = nn.Sequential(
            nn.Linear(250,512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.ReLU(),

            nn.Linear(1024, 33443)
        )

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded, encoded 

def train_AE(seed = 42, batch_size = 32, lr = 1e-3, decay = 1e-5, epochs=100,
            noise_factor = 0.1):
    data, validation, _ = load_standardized_data()
    without_labels = data.drop(columns=['BRCA_Subtype_PAM50'])
    val = validation.drop(columns=['BRCA_Subtype_PAM50'])
    data_tensor = torch.tensor(without_labels.values, dtype=torch.float32)
    val_tensor = torch.tensor(val.values, dtype = torch.float32)
    dataset = TensorDataset(data_tensor)
    valset = TensorDataset(val_tensor)

    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    train_loader = DataLoader(
        dataset,
        batch_size = batch_size, 
        shuffle = True
    )

    val_loader = DataLoader(
        valset, 
        batch_size = batch_size,
        shuffle = False
    )
    
    model = AE().to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = decay)

    losses = []
    val_losses = []
    patience = 15
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for (batch, ) in train_loader:
            batch = batch.to(device)
            noise_batch = batch + noise_factor*torch.rand_like(batch)
            optimizer.zero_grad()
            recon, _ = model(noise_batch)
            loss = loss_function(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        losses.append(total_loss/len(train_loader))
        if epoch % 10 == 0:
            print(f'Epoch {epoch} loss: {total_loss/len(train_loader)}')

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = loss_function(recon, batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            epochs_no_improve = 0 
            best_val_loss = val_loss
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after training for {epoch} epochs')
            print(f'Final training MSE loss: {total_loss/len(train_loader)}')
            print(f'Final validation MSE loss: {val_loss}')
            break


    torch.save(model.state_dict(), 'tcga-autoencoder.pt')

    plt.figure()
    plt.plot(losses, label = 'train loss')
    plt.plot(val_losses, label = 'validation loss')
    plt.xlabel('epoch')
    plt.ylabel('average loss')
    plt.title('training loss over time')
    plt.legend()
    plt.show()

    return model

def write_AE_data():
    model = AE()
    model.load_state_dict(torch.load('tcga-autoencoder.pt'))
    model.eval()

    train, val, test = load_standardized_data()
    data_train = train.drop(columns='BRCA_Subtype_PAM50')
    label_train = train.loc[:,'BRCA_Subtype_PAM50']
    data_val = val.drop(columns='BRCA_Subtype_PAM50')
    label_val = val.loc[:,'BRCA_Subtype_PAM50']
    data_test = test.drop(columns='BRCA_Subtype_PAM50')
    label_test= test.loc[:,'BRCA_Subtype_PAM50']


    train_tensor = torch.tensor(data_train.values, dtype = torch.float32)
    val_tensor = torch.tensor(data_val.values, dtype = torch.float32)
    test_tensor = torch.tensor(data_test.values, dtype = torch.float32)

    with torch.no_grad():
        _, train_encode = model(train_tensor)
        _, val_encode = model(val_tensor)
        _, test_encode = model(test_tensor)

    train_encoded = pd.DataFrame(train_encode.detach().numpy())
    val_encoded = pd.DataFrame(val_encode.detach().numpy())
    test_encoded = pd.DataFrame(test_encode.detach().numpy())

    train_ae = pd.concat(
        [label_train.reset_index(drop = True), 
        train_encoded.reset_index(drop=True)], axis = 1)
    val_ae = pd.concat(
        [label_val.reset_index(drop=True),
         val_encoded.reset_index(drop=True)], axis = 1)
    test_ae = pd.concat(
        [label_test.reset_index(drop=True),
         test_encoded.reset_index(drop=True)], axis = 1)

    train_ae.to_feather('tcga-brca-ae-train.feather')
    val_ae.to_feather('tcga-brca-ae-val.feather')
    test_ae.to_feather('tcga-brca-ae-test.feather')

    return train_ae, val_ae, test_ae

def load_ae_data():
    train = pd.read_feather('tcga-brca-ae-train.feather')
    val = pd.read_feather('tcga-brca-ae-val.feather')
    test = pd.read_feather('tcga-brca-ae-test.feather')

    return train, val, test


main()
import pandas as pd 
import numpy as np
import seaborn as sns
import os 
import torch

from torch.utils.data import DataLoader

#MNIST DATASET

def data_preprocess(path_to_train, path_to_test):
   #path_train = "" #path to dataset
   #path_test = "" #path to dataset 

   train_data_raw = pd.read_csv(path_to_train, sep=",")
   test_data_raw = pd.read_csv(path_to_test, sep=",")

   train_labels = train_data_raw['label']
   train_data_raw.drop('label', axis=1, inplace=True) # remove label column

   test_labels = test_data_raw['label']
   test_data_raw.drop('label', axis=1, inplace=True) # remove label column

   train_data = train_data_raw.values
   train_labels = train_labels.values

   test_data = test_data_raw.values
   test_labels = test_labels.values

   dim = 28

   reshaped_train = []
   for i in train_data:
       reshaped_train.append(i.reshape(1, dim, dim))
   train_data = np.array(reshaped_train)

   reshaped_test = []
   for i in test_data:
      reshaped_test.append(i.reshape(1, dim, dim))
   test_data = np.array(reshaped_train)

   train_x = torch.FloatTensor(train_data)
   train_y = torch.LongTensor(train_labels.tolist())

   test_x = torch.FloatTensor(train_data)
   test_y = torch.LongTensor(train_labels.tolist())

   #train_data_loader = DataLoader(train_data, batch_size = 100, num_workers = 2, shuffle = True)
   #test_data_loader = DataLoader(test_data, batch_size = 100, num_workers = 0)

   #return train_data_loader, test_data_loader #--> gestire con dataloader alla fine 
   return train_x, train_y, test_x, test_y
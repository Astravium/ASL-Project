import numpy as np
import torch
from utils.dataProcessing import data_preprocess


from model.model_def_v1 import Network                         # <--- import model 
from model.modelTraining import train_model_v1                 # <--- import model training


def main():

   #app principale con ui etc... 

   #PATH TO DATA
   path_to_train = "DatasetMNIST/sign_mnist_train/sign_mnist_train.csv"
   path_to_test = "DatasetMNIST/sign_mnist_test/sign_mnist_test.csv" 


   #DATA PREPROCESSING
   train_x, train_y, test_x, test_y = data_preprocess(path_to_train, path_to_test) 

   #MODEL INSTANCE
   use_cuda = torch.cuda.is_available()
   if use_cuda:
      model = Network().cuda()
      train_x = train_x.cuda()
      train_y = train_y.cuda()
      test_x = test_x.cuda()
      test_y = test_y.cuda()
   else:
      model = Network()


   #MODEL TRAINING
   train_model_v1(model, train_x, train_y, test_x, test_y, use_cuda,lr=0.0001,momentum=0.90, epochs=50)

   #MODEL EVAL+TEST
   predictions = model(test_x)
   accuracy, correct, total = model.test(torch.max(predictions.data, 1)[1], test_y)
   print("Accuracy = " + str(accuracy) + " ("+str(correct)+"/"+str(total)+")")
   
   #SAVE MODEL
   torch.save(model, 'model_v1_trained.pt')
   
main()
import torch
import torch.nn as nn
import torch.nn.functional as F

""" class Network(nn.Module):

   def __init__(self):
       
       super(Network, self).__init__()

       self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
       self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
       self.pool1 = nn.MaxPool2d(2)
       self.dropout1 = nn.Dropout2d(0.25)

       self.conv3 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
       self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
       self.pool2 = nn.MaxPool2d(2)
       self.dropout2 = nn.Dropout2d(0.25)

       self.conv5 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
       self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
       self.pool3 = nn.MaxPool2d(2)
       self.dropout3 = nn.Dropout2d(0.25)


       self.flatten1 = nn.Flatten()
       self.fc1 = nn.Linear()
       self.dropout4 = nn.Dropout2d(0.5)

       self.fc2 = nn.Linear()
       self.dropout5 = nn.Dropout2d(0.5)

       self.fc2 = nn.Linear()
       self.softmax = nn.LogSoftmax(dim=1)


   def forward(self,x): """
       
class MyModel(nn.Module):
    def __init__(self, input_shape):
        super(MyModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        
        # Assuming the input image size is known and is the same as in Keras input_shape (H, W, C)
        # You need to calculate the flattened dimension after all convolutions and poolings
        # Example: for an input size of (128, 128, 3) it would be:
        # self.flattened_dim = 64 * (input_height // 8) * (input_width // 8)
        # In this example, it is:
        self.flattened_dim = 64 * (input_shape[1] // 8) * (input_shape[2] // 8)

        self.fc1 = nn.Linear(self.flattened_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 22)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        
        return x
    
    def test(self, predictions,labels):
        self.eval()
        correct = 0
        for p,l in zip(predictions,labels):
            if p==l:
                correct+=1
        acc = correct/len(predictions)
        return(acc, correct, len(predictions))
    
    def evaluate(self, predictions,labels):
        correct = 0
        for p,l in zip(predictions,labels):
            if p==l:
                correct+=1
        acc = correct/len(predictions)
        return(acc)
        


""" 
def istantiate_model(input_shape, print_bool = True):
    model = Sequential()
    #1st
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same',input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #2nd
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same',input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #3d
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same',input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #4th
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))

    #5th
    model.add(Dense(128))
    model.add(Dropout(0.5))

    #6th
    model.add(Dense(22))
    model.add(Activation('softmax'))

    if (print_bool != False):
        model.summary()

    return model """
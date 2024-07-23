import torch
import torch.nn as nn
import torch.nn.functional as F

class SignLanguageCNN(nn.Module):
    def __init__(self):
        super(SignLanguageCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, stride=2)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_fc = nn.Dropout(0.15)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 4 * 4, 1024) # 64* 4 * 4 = 1024
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 22)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten the tensor while preserving the batch size
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        # last layer, use softmax
        x = self.softmax(self.fc3(x))

        return x

    def test(self, predictions, labels):
        self.eval()
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
        acc = correct / len(predictions)
        return acc, correct, len(predictions)

    def evaluate(self, predictions, labels):
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
            acc = correct / len(predictions)
        return acc


class SignLanguageCNN_V2(nn.Module):
    def __init__(self):
        super(SignLanguageCNN_V2, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_fc = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 22)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten the tensor while preserving the batch size
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.softmax(self.fc3(x))

        return x


    def test(self, predictions, labels):
        self.eval()
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
        acc = correct / len(predictions)
        return acc, correct, len(predictions)

    def evaluate(self, predictions, labels):
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
            acc = correct / len(predictions)
        return acc

class SignLanguageCNN_india(nn.Module):
    def __init__(self):
        super(SignLanguageCNN_india, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # input shape (1, 64, 64), output shape (16, 64, 64)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) #
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.dropout_conv = nn.Dropout2d(0.25)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_fc = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 4 * 4, 512) # input shape (64, 4, 4), output shape (512)
        self.fc2 = nn.Linear(512, 22) # input shape (512), output shape (22)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # input shape (1, 64, 64), output shape (16, 32, 32)
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.conv2(x))) # input shape (16, 32, 32), output shape (32, 16, 16)
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.conv3(x))) # input shape (32, 16, 16), output shape (64, 8, 8)
        x = self.dropout_conv(x)
        
        x = self.pool(x) # input shape (64, 8, 8), output shape (64, 4, 4)

        x = x.view(x.size(0), -1)  # Flatten the tensor while preserving the batch size
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.softmax(x)

        return x


    def test(self, predictions, labels):
        self.eval()
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
        acc = correct / len(predictions)
        return acc, correct, len(predictions)

    def evaluate(self, predictions, labels):
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
            acc = correct / len(predictions)
        return acc


class SignLanguage_DeepCNN(nn.Module):
    def __init__(self):
        super(SignLanguage_DeepCNN, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), #input: 1x64x64, output: 16x64x64
            nn.ReLU(),

            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # input: 16x64x64, output: 32x64x64
            nn.MaxPool2d(2, 2), #input: 32x64x64, output: 32x32x32
            nn.ReLU(),

            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), #input: 32x32x32, output: 64x32x32
            nn.MaxPool2d(2, 2),  # input: 64x32x32, output: 64x16x16
            nn.ReLU(),

            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), #input: 64x16x16, output: 64x16x16
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),

            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #input: 64x16x16, output: 128x16x16
            nn.MaxPool2d(2, 2), #input: 128x16x16, output: 128x8x8
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(128 * 8 * 8, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 22, bias=True),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.classifier(x.view(x.shape[0],-1))
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


import torch.nn as nn
import torch.nn.functional as F

class Network_64x64(nn.Module):
    def __init__(self):
        super(Network_64x64, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv3 = nn.Conv2d(20, 30, 3)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.4)

        # Calculate the size after convolution and pooling layers
        # Input: 64x64 -> Conv1 -> 62x62 -> Pool -> 31x31
        # 31x31 -> Conv2 -> 29x29 -> Pool -> 14x14
        # 14x14 -> Conv3 -> 12x12
        self.fc1 = nn.Linear(30 * 12 * 12, 270)
        self.fc2 = nn.Linear(270, 22)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = x.view(-1, 30 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.softmax(F.relu(self.fc2(x)))

        return x

class Network_64x64_RGB(nn.Module):
    def __init__(self):
        super(Network_64x64_RGB, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv3 = nn.Conv2d(20, 30, 3)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.4)

        # Calculate the size after convolution and pooling layers
        # Input: 64x64 -> Conv1 -> 62x62 -> Pool -> 31x31
        # 31x31 -> Conv2 -> 29x29 -> Pool -> 14x14
        # 14x14 -> Conv3 -> 12x12
        self.fc1 = nn.Linear(30 * 12 * 12, 270)
        self.fc2 = nn.Linear(270, 22)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = x.view(-1, 30 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.softmax(F.relu(self.fc2(x)))

        return x


class Network_28x28(nn.Module):
    def __init__(self):
        super(Network_28x28, self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,20,3)
        self.conv3 = nn.Conv2d(20,30,3)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.2)
        
        self.fc1 = nn.Linear(30*3*3, 270)
        self.fc2 = nn.Linear(270,26)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x.view(-1, 30 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.softmax(F.relu(self.fc2(x)))
        
        return(x)
        
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


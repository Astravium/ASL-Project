import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np

class AverageValueMeter():
   
   def __init__(self):
      self.reset()

   def reset(self):
      self.sum=0
      self.num=0

   def add(self, value, num):
      self.sum += value*num
      self.num += num

   def value(self):
      try:
         return self.sum/self.num
      except:
         return None



def train_model_lab(model, train_data, test_data, lr=0.01, epochs=10, momentum=0.99):
   
   criterion = nn.CrossEnrtropyLoss()
   optimizer = SDG(model.parameters(), lr, momentum=momentum)

   loss_meter = AverageValueMeter()
   acc_meter = AverageValueMeter()

   writer = SummaryWriter(join(logdir, exp_name))

   device = "cuda" if torch.cuda.is_available() else "cpu"
   model.to(device)

   loader = {
      'train' : train_data,
      'test' : test_data
   } 

   global_step = 0

   for e in range(epochs):

      for mode in ['train','test']:
         loss_meter.reset()
         acc_meter.reset()

         with torch.set_grad_enabled(mode=="train"):
            for i,batch in enumerate(loader[mode]):
               x=batch[0].to(device)
               y=batch[1].to(device)
               output = model(x)

               n = x.shape[0]
               global_step += n
               l = criterion(output,y)

               if mode=='train':
                  l.backward()
                  optimizer.step()
                  optimizer.zero_grad()

               acc = accuracy_score(y.to('cpu'), output.to('cpu').max(1)[1])
               loss_meter.add(l.item(),n)
               acc_meter.add(acc,n)

               if mode=='train':
                  writer.add_scalar('loss/train', loss_meter.value(), global_step=global_step)
                  writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)
         
         writer.add_scalar('loss/' + mode, loss_meter.value(), global_step=global_step)
         writer.add_scalar('accuracy/' + mode, acc_meter.value(), global_step=global_step)

      torch.save(mode.state_dict(), '%s-%d.pth' % (exp_name, e+1))
   
   return model

def train_model_v1(model, train_x, train_y, test_x, test_y, use_cuda, lr=0.01, momentum=0.90, epochs=50):

   optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
   criterion = nn.CrossEntropyLoss()

   loss_log = []
   acc_log = []
   
   model.to("cuda" if use_cuda else "cpu")

   for e in range(epochs):
      for i in range(0, train_x.shape[0], 64):
         y_mini = train_y[i:i + 64] 
         x_mini = train_x[i:i + 64] 
         
         if use_cuda:
               x_mini = x_mini.cuda()
               y_mini = y_mini.cuda()
               test_x = test_x.cuda()
               test_y = test_y.cuda()
         
         optimizer.zero_grad()
         #print(x_mini.shape)
         net_out = model(Variable(x_mini))
         
         loss = criterion(net_out,y_mini)
         loss.backward()
         optimizer.step()
         
         if i % 1000 == 0:
               #pred = net(Variable(test_data_formated))
               loss_log.append(loss.item())
               acc_log.append(model.evaluate(torch.max(model(Variable(test_x[:500])).data, 1)[1], test_y[:500]))
         
      print('Epoch: {} - Loss: {:.6f}'.format(e + 1, loss.item()))
   
   plt.figure(figsize=(10,8))
   plt.plot(loss_log[2:])
   plt.plot(acc_log)
   plt.plot(np.ones(len(acc_log)), linestyle='dashed')
   plt.show()
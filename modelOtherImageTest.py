import numpy as np
import torch

model = torch.load('model_v1_trained.pt')
model.eval()

# load test image
from PIL import Image
from torchvision import transforms

alph = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
        18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'}

print(len(alph))

# open image in grayscale
img = Image.open('img_MNIST_dataset/img_151.png').convert('L')
# img = Image.open('img_SELF_dataset/O_elvio.jpg').convert('L')
img = img.resize((28, 28))
img.show()

# send data to gpu

data = transforms.ToTensor()(img)
model_input = data.reshape(1, 1, 28, 28)
model_input = model_input.cuda()
output = model(model_input)
print(output)

# print prediction based on alph
print("Prediction: ", alph[output.argmax().item()])
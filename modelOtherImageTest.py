import numpy as np
import torch

model = torch.load('SignLanguageCNN_lr_0001_mom99_ep15.pt')
model.eval()

# load test image
from PIL import Image
from torchvision import transforms

alph = {
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f',
        6: 'h',
        7: 'i',
        8: 'k',
        9: 'l',
        10: 'm',
        11: 'n',
        12: 'o',
        13: 'p',
        14: 'q',
        15: 'r',
        16: 't',
        17: 'u',
        18: 'v',
        19: 'w',
        20: 'x',
        21: 'y',
}

print(len(alph))

# open image in grayscale
img = Image.open('DataAcquisition/pythonProject/Dataset_Elvio/o/o_Elvio_142.jpg')
# img = Image.open('img_SELF_dataset/O_elvio.jpg').convert('L')
# img = img.resize((28, 28))
img.show()

# send data to gpu

data = transforms.ToTensor()(img)
model_input = data.reshape(1, 3, 64, 64)
model_input = model_input.cuda()
output = model(model_input)
print(output)


# print prediction based on alph
print("Prediction: ", alph[output.argmax().item()])
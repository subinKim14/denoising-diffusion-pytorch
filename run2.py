import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)


EXTS = ['jpg', 'jpeg', 'png']
folder = './test'
image_size = 128
paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]

print(len(paths))
data = []
for i in range(len(paths)):
    #path = paths[i]
    img = cv2.imread('./test/test'+str(i+1)+'.png', cv2.IMREAD_COLOR)
    img = img.astype('float64')
    img = (img-127)/255
    img2 = np.swapaxes(img, 0, 2)
    data.append(img2)
    data.append(img2)

data = np.array(data)
print(data.shape)
print('max: ', np.min(data))
print('min: ', np.max(data))

training_images = torch.from_numpy(data)
training_images = training_images.type(torch.FloatTensor)
#utils.save_image(training_images, str('result_training.png'), nrow=4)
print(training_images.type())


loss = diffusion(training_images)
loss.backward()
# after a lot of training

sampled_images = diffusion.sample(128, batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)

#utils.save_image(sampled_images, str('result_sample.png'), nrow=4)

#sampled_images = sampled_images.type(torch.ByteTensor)
sampled_data = sampled_images.numpy()

print('max: ', np.min(sampled_data))
print('min: ', np.max(sampled_data))

sampled_data = sampled_data*255+127
sampled_data = sampled_data.astype('uint8')
print('max: ', np.min(sampled_data))
print('min: ', np.max(sampled_data))

for i in range(len(paths)):
    #path = paths[i]
    img = sampled_data[i]
    img = np.swapaxes(img, 0, 2)
    cv2.imwrite('./test/sampled'+str(i+1)+'.png', img)
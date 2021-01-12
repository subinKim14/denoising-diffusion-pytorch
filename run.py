from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch, gc
gc.collect()
torch.cuda.empty_cache()
CUDA_VISIBLE_DEVICES=0
device_num = 0
torch.cuda.set_device(device_num)
print(torch.cuda.current_device())

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    './data256x256',
    image_size = 256,
    train_batch_size = 4,
    train_lr = 2e-5,
    train_num_steps = 500000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.9999,                # exponential moving average decay
    fp16 = True                      # turn on mixed precision training with apex
)

trainer.train()
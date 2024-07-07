#!/usr/bin/env python
# coding: utf-8
import diffusers
import torch
from tqdm.autonotebook import tqdm
import torchvision
import torch
from torchvision.transforms import v2
import wandb
import json
import argparse
import os

wandb.login()

def get_dataset(dataset_type, data_dir):
    if dataset_type == "mnist":    
        transform = v2.Compose(
            [v2.ToImage(),
             v2.ToDtype(torch.float32, scale=True),
             v2.ToPureTensor(),
             v2.Normalize(mean=[0.5], std=[0.5])], 
            )
        dataset = torchvision.datasets.MNIST(data_dir, download=True, transform=transform)
        return dataset
    elif dataset_type == "fashion-mnist":
        transform = v2.Compose(
            [v2.ToImage(),
             v2.RandomHorizontalFlip(0.5),
             v2.ToDtype(torch.float32, scale=True),
             v2.ToPureTensor(),
             v2.Normalize(mean=[0.5], std=[0.5])] 
            )
        dataset = torchvision.datasets.FashionMNIST(data_dir, download=True, transform=transform)
        return dataset
    elif dataset_type == "cifar-10":
        transform = v2.Compose(
            [v2.ToImage(),
             v2.RandomHorizontalFlip(0.5),
             v2.ToDtype(torch.float32, scale=True),
             v2.ToPureTensor(),
             v2.Normalize(mean=[0.5], std=[0.5])], 
            )
        dataset = torchvision.datasets.CIFAR10(data_dir, download=True, transform=transform)
        return dataset
    elif dataset_type == "imagenet-32":
        from imagenet import ImageNetDownSample
        transform = v2.Compose(
            [v2.ToImage(),
             v2.RandomHorizontalFlip(0.5),
             v2.ToDtype(torch.float32, scale=True),
             v2.ToPureTensor(),
             v2.Normalize(mean=[0.5], std=[0.5])], 
            )
        dataset = ImageNetDownSample(os.path.join(data_dir, "Imagenet32"), transform=transform)
        return dataset
    else:
        assert False

@torch.no_grad
def sample_image(model, max_timestep, image_shape, batch_size, class_labels, use_classlabel, device):
    X0 = torch.randn((batch_size, *image_shape)).to(device)
    for i in range(max_timestep):
        timestep = torch.full(size=(batch_size,), fill_value=i).to(device)
        if use_classlabel:
            pred = model(X0, timestep=timestep, class_labels = class_labels)
        else:
            pred = model(X0, timestep=timestep)
        X0 += pred.sample * (1/max_timestep)
    return X0

def train_rectified_flow(model, 
                         optimizer, 
                         dataloader, 
                         max_class = None,
                         use_classlabel = False,
                         max_timestep = 1000,
                         epoch = 16, 
                         device = None,
                         dtype = torch.float16):
    model = model.to(device)
    if use_classlabel:
        assert max_class is not None
    loss_fun = torch.nn.MSELoss()
    def normalize_image(X):
        X = torchvision.utils.make_grid(X)
        X = (X + 1)/2
        return X
    def eval_image():
        batch_size = 64
        class_labels = torch.arange(batch_size).to(device) % max_class 
        image_array = sample_image(model, 
                                   max_timestep, 
                                   image_shape = (IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH), 
                                   batch_size = batch_size, 
                                   class_labels = class_labels,
                                   use_classlabel = use_classlabel,
                                   device= device)
        image_array = normalize_image(image_array).permute((1,2,0)).cpu().detach().numpy()
        images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")
        wandb.log({"examples": images}, commit=False)
        
    scaler = torch.GradScaler()    
    for i in range(epoch):
        if i == 0:
            eval_image()
        wandb.log({"epoch": i}, commit=False)
        for (image, class_labels) in dataloader:
            # image : shape (b, c, h, w)
            optimizer.zero_grad(set_to_none = True)
            image = image.to(device)
            class_labels = class_labels.to(device)
            noise = torch.randn(image.shape).to(device)
            assert image.dtype == noise.dtype
            timestep = torch.randint(high = max_timestep, size=(image.size(0),)).to(device)
            alpha = (timestep / max_timestep).view(-1, *([1]*(len(image.shape) - 1)))
            point = (1 - alpha) * noise + alpha * image
            with torch.autocast(enabled=(dtype != torch.float32), device_type=device, dtype=dtype):
                if use_classlabel:
                    pred = model(point, timestep=timestep, class_labels = class_labels)
                else:
                    pred = model(point, timestep=timestep)
                target = image - noise
                loss = loss_fun(pred.sample, target)
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()
            grad_norm = torch.sqrt(sum([torch.norm(p.grad)**2 for p in model.parameters()]))
            wandb.log({"grad_scale" : scaler.get_scale()}, commit=False)
            wandb.log({"grad_norm": grad_norm.item()}, commit = False)
            wandb.log({"loss": loss.item()}, commit=True)
        eval_image()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type = str)
parser.add_argument('--data_dir', type = str)
args = parser.parse_args()
with open(args.config) as f:
    config = json.load(f)

lr = config.get("lr", 1e-4)
batch_size = config.get("batch_size", 256)
use_fake = config.get("use_fake", False)
epochs = config["epochs"]
max_timestep = config["max_timestep"]
dtype = config["dtype"]
dtypemap = {
    "fp16" : torch.float16,
    "bf16" : torch.bfloat16,
    "fp32" : torch.float32
}
dtype = dtypemap[dtype]
dataset_type = config["dataset_type"]
data_dir = args.data_dir
dataset = get_dataset(dataset_type, data_dir)
IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH = dataset[0][0].shape
num_embeds_ada_norm = len(dataset.classes)
model_config = config["model"]
print(model_config)
model = diffusers.DiTTransformer2DModel(
    **model_config
)
print("model size:", sum([p.numel() for p in model.parameters()]))

def get_dtype_str(x):
    if x == torch.float16:
        return "fp16"
    elif x == torch.bfloat16:
        return "bf16"
    elif x == torch.float32:
        return "fp32"
    else:
        assert False

run = wandb.init(
    # Set the project where this run will be logged
    project="dit-rectified-flow",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
        "dit_config": model.config,
        "batch_size" : batch_size,
        "epochs" : epochs,
        "max_timestep" : max_timestep,
        "use_classlabel" : True,
        "dataset" : str(dataset),
        "dataset_type" : dataset_type,
        "dtype" : str(dtype),
    },
    name = config["run"]
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

def make_dataloader(*args, **kwargs):
    if use_fake:
        def fake():
            while True:
                yield 0
        dataloader = torch.utils.data.DataLoader(*args, **kwargs, sampler = fake())
    else:
        dataloader = torch.utils.data.DataLoader(*args, **kwargs)
    return dataloader
dataloader = make_dataloader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

train_rectified_flow(model, 
                     optimizer, 
                     dataloader, 
                     use_classlabel=True, 
                     max_class = len(dataset.classes),
                     max_timestep=max_timestep, epoch=epochs,
                     device="cuda",
                     dtype = dtype)

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

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
from timer import Timer

# === data ===
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

def make_dataloader(*args, use_fake=False, **kwargs):
    if use_fake:
        def fake():
            while True:
                yield 0
        dataloader = torch.utils.data.DataLoader(*args, **kwargs, sampler = fake())
    else:
        dataloader = torch.utils.data.DataLoader(*args, **kwargs)
    return dataloader

# === training and evaluation ===
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

def train_rectified_flow(rank : int,
                         world_size : int,
                         model, 
                         optimizer, 
                         dataloader,
                         sampler, 
                         image_shape,
                         max_class = None,
                         use_classlabel = False,
                         max_timestep = 1000,
                         epoch = 16, 
                         dtype = torch.float16):
    device = rank
    if use_classlabel:
        assert max_class is not None
    loss_fun = torch.nn.MSELoss()
    def normalize_image(X):
        X = torchvision.utils.make_grid(X)
        X = (X + 1)/2
        return X
    def eval_image():
        model.eval()
        assert world_size < 64 and 64 % world_size == 0
        batch_size = 64 // world_size
        class_labels = torch.arange(batch_size, device=rank) % max_class 
        image_array = sample_image(model, 
                                   max_timestep, 
                                   image_shape = image_shape, 
                                   batch_size = batch_size, 
                                   class_labels = class_labels,
                                   use_classlabel = use_classlabel,
                                   device= device)
        
        image_arrays = [torch.zeros_like(image_array, device=rank) for _ in range(world_size)]
        dist.all_gather(image_arrays, image_array)
        if rank == 0:
            image_array = torch.cat(image_arrays, dim = 0)
            image_array = normalize_image(image_array).permute((1,2,0)).cpu().detach().numpy()
            images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")
            wandb.log({"examples": images}, commit=False)

    timer = Timer()      
    scaler = torch.GradScaler()    

    for i in range(epoch):
        if i == 0:
            eval_image()
        # reset sampler
        if sampler is not None:
            sampler.set_epoch(i)
        if rank == 0:
            wandb.log({"epoch": i}, commit=False)
        for (image, class_labels) in dataloader:
            model.train()
            # image : shape (b, c, h, w)
            optimizer.zero_grad(set_to_none = True)
            image = image.to(device)
            class_labels = class_labels.to(device)
            noise = torch.randn(image.shape).to(device)
            assert image.dtype == noise.dtype
            timestep = torch.randint(high = max_timestep, size=(image.size(0),)).to(device)
            alpha = (timestep / max_timestep).view(-1, *([1]*(len(image.shape) - 1)))
            point = (1 - alpha) * noise + alpha * image

            avg_values = torch.zeros((1,), device=rank)
            with torch.autocast(enabled=(dtype != torch.float32), device_type="cuda", dtype=dtype):
                if use_classlabel:
                    pred = model(point, timestep=timestep, class_labels = class_labels)
                else:
                    pred = model(point, timestep=timestep)
                target = image - noise
                loss = loss_fun(pred.sample, target)
                avg_values[0] = loss.item()
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()
            dist.all_reduce(avg_values, op=dist.ReduceOp.AVG)
            if rank == 0:
                timer.step()
                wandb.log({"sample_per_second" : world_size * dataloader.batch_size * timer.rate()}, commit=False)
                wandb.log({"grad_scale" : scaler.get_scale()}, commit=False)
                wandb.log({"grad_norm": grad_norm.item()}, commit = False)
                wandb.log({"loss": avg_values[0].item()}, commit=True)
        eval_image()

# === util ===
def get_dtype_str(x):
    if x == torch.float16:
        return "fp16"
    elif x == torch.bfloat16:
        return "bf16"
    elif x == torch.float32:
        return "fp32"
    else:
        assert False

# === distributed === 
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Define a closure here, so all the variables above are accessible here.... 
def main(rank, world_size, args, config):
    setup(rank, world_size)
    # Setting model configuration
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
    image_shape = dataset[0][0].shape
    # num_embeds_ada_norm = len(dataset.classes)
    model_config = config["model"]
    # we only call wandb in the first process
    if rank == 0:
        print(model_config)
        wandb.login()
        wandb.init(
            # Set the project where this run will be logged
            project="dit-rectified-flow",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "epochs": epochs,
                "dit_config": model_config,
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
    model = diffusers.DiTTransformer2DModel(**model_config).to(rank)
    model = DDP(model)
    if rank == 0:
        print("model size:", sum([p.numel() for p in model.parameters()]))
    optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr = lr, weight_decay=0.0)
    if world_size == 1:
        sampler = None
        dataloader = make_dataloader(dataset, use_fake=use_fake, batch_size=batch_size, drop_last=True, shuffle=True)
    else:
        assert batch_size % world_size == 0
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed = 0, drop_last=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size // world_size, sampler=sampler)
    train_rectified_flow(
                    rank,
                    world_size,
                    model, 
                    optimizer, 
                    dataloader, 
                    sampler,
                    image_shape,
                    use_classlabel=True, 
                    max_class = len(dataset.classes),
                    max_timestep=max_timestep, epoch=epochs,
                    dtype = dtype)
    if rank == 0:
        wandb.finish()
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str)
    parser.add_argument('--data_dir', type = str)
    parser.add_argument('--ddp', type=int)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    world_size = args.ddp
    if torch.cuda.device_count() != world_size:
        raise BaseException(f"DDP size {world_size} must be equal to the visiable device number {torch.cuda.device_count()} !")
    
    # === reproducibility ===
    import random
    random.seed(0)
    import numpy as np
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    mp.spawn(main, args=(world_size, args, config), nprocs=world_size, join=True)

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
import numpy as np
from typing import Any, Tuple, List

from streaming.vision.base import StreamingDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
import streaming
from dataclasses import dataclass

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

@dataclass
class ImageDatasetInfo:
    dataset : Any
    is_streaming_dataset : bool
    image_shape : Tuple[int, int, int]
    classes : List[str]

    is_latent : bool
    vae : Any
    vae_preprocessor : Any
    vae_postprocessor : Any
    class_condition : bool

# === data ===
def get_dataset(rank : int, dataset_type, data_dir, temp_dir, batch_size : int) -> ImageDatasetInfo:
    # the batch_size here is per device batch_size
    image_shape = None
    is_streaming_dataset = False
    classes = None
    is_latent = False
    vae = None
    vae_preprocessor = None
    vae_postprocessor = None
    class_condition = True
    if dataset_type == "mnist":    
        transform = v2.Compose(
            [v2.ToImage(),
             v2.ToDtype(torch.float32, scale=True),
             v2.ToPureTensor(),
             v2.Normalize(mean=[0.5], std=[0.5])], 
            )
        dataset = torchvision.datasets.MNIST(data_dir, download=True, transform=transform)
    elif dataset_type == "fashion-mnist":
        transform = v2.Compose(
            [v2.ToImage(),
             v2.RandomHorizontalFlip(0.5),
             v2.ToDtype(torch.float32, scale=True),
             v2.ToPureTensor(),
             v2.Normalize(mean=[0.5], std=[0.5])] 
            )
        dataset = torchvision.datasets.FashionMNIST(data_dir, download=True, transform=transform)
    elif dataset_type == "cifar-10":
        transform = v2.Compose(
            [v2.ToImage(),
             v2.RandomHorizontalFlip(0.5),
             v2.ToDtype(torch.float32, scale=True),
             v2.ToPureTensor(),
             v2.Normalize(mean=[0.5], std=[0.5])], 
            )
        dataset = torchvision.datasets.CIFAR10(data_dir, download=True, transform=transform)
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
    elif dataset_type == 'imagenet.uint8':
        is_streaming_dataset = True
        image_shape = (4, 32, 32)
        is_latent = True  
        class uint8(Encoding):
            def encode(self, obj: Any) -> bytes:
                return obj.tobytes()

            def decode(self, data: bytes) -> Any:
                x=  np.frombuffer(data, np.uint8).astype(np.float32)
                return (x / 255.0 - 0.5) * 24.0
        _encodings['uint8'] = uint8
        # we must use the same temporary data directory here
        local = temp_dir
        remote = os.path.join(data_dir, "vae_mds")
        # we should use this new scaling factor
        # the variance of input now will be 1
        scaling_factor = 1 / 0.13025
        class DatasetAdaptor(StreamingDataset):
            def __init__(self,
                        *args, 
                        **kwargs
                        ) -> None:
                super().__init__(*args, **kwargs)
            def __getitem__(self, idx:int):
                obj = super().__getitem__(idx)
                x = obj['vae_output']
                y = obj['label']
                return x.reshape(4, 32, 32) / scaling_factor, int(y)
        # the raw data is a tensor with value in range [-12, 12]
        # this is difficult for the model to learn, so we firstly scale the tensor into [-1, 1]
        # then we rescale the learnt model into [-12, 12] before sampling
        dataset = DatasetAdaptor(
            local=local, 
            remote=remote, 
            split=None,
            shuffle=True,
            shuffle_algo="naive",
            num_canonical_nodes=1,
            batch_size = batch_size)
        classes = [str(i) for i in range(1000)]
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(rank)
        vae_preprocessor = lambda x : x * scaling_factor
        vae = vae.eval()
        vae_postprocessor = VaeImageProcessor(do_normalize=True)
    elif dataset_type == "mj":
        is_streaming_dataset = False
        image_shape = (4, 32, 32)
        is_latent = True  
        class_condition = False
        from torch import Tensor
        """
        Convert latent to unit variance
        """
        class LatentProcessor:
            def __init__(self):
                self.mean = 0.8965
                self.scale_factor = 0.13025
            @torch.no_grad
            def preprocess(self, latents : Tensor):
                latents = (latents - self.mean) * self.scale_factor
                return latents.float()
            @torch.no_grad
            def postprocess(self, latents : Tensor):
                assert isinstance(latents, Tensor)
                latents = self.mean + (latents / self.scale_factor)
                # the vae is in fp16 so...
                return latents.half()
            def test(self, latents):
                return (self.postprocess(self.preprocess(latents)) - latents).abs().mean()

        latent_processor = LatentProcessor()
        class DatasetAdaptor(torch.utils.data.Dataset):
            def __init__(self) -> None:
                self.latents = np.load(os.path.join(data_dir, "small_ldt/mj_latents.npy"), mmap_mode='r') 
                self.text_emb = np.load(os.path.join(data_dir, "small_ldt/mj_text_emb.npy"), mmap_mode='r') 
            def __len__(self):
                return len(self.latents)
            def __getitem__(self, idx:int):
                x = latent_processor.preprocess(torch.tensor(self.latents[idx], requires_grad=False))
                y = torch.tensor(self.text_emb[idx], requires_grad=False)
                return x, y
        class ImageProcessor:
            def postprocess(self, image, *args, **kwargs):
                return (image + 1) / 2
        dataset = DatasetAdaptor()
        # unconditional generation, we only use one class
        classes = [str(i) for i in range(1)]
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(rank)
        vae = vae.eval()
        # covert unit variance latent back to unscaled latent
        vae_preprocessor = latent_processor.postprocess
        # convert [-1, 1] image to [0, 1] scaled
        vae_postprocessor = ImageProcessor()
    if image_shape is None:
        image_shape = dataset[0][0].shape
    if classes is None:
        classes = dataset.classes
    return ImageDatasetInfo(dataset, is_streaming_dataset, image_shape, classes, is_latent, vae, vae_preprocessor, vae_postprocessor, class_condition)

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
def sample_image(rank : int, model, max_timestep: int, sampling_step : int, batch_size, class_labels, datasetinfo : ImageDatasetInfo, use_classlabel, device):
    image_shape = datasetinfo.image_shape
    # we add a generator to better monitor the quality here
    X0 = torch.randn((batch_size, *image_shape)).to(device)
    for i in range(sampling_step):
        val = (i / sampling_step) * max_timestep
        # we use a uniform sampling here
        timestep = torch.full(size=(batch_size,), fill_value=val).to(device)
        if use_classlabel:
            pred = model(X0, timestep=timestep, class_labels = class_labels)
        else:
            pred = model(X0, timestep=timestep)
        X0 += pred.sample * (1 / sampling_step)
    if datasetinfo.is_latent:
        X0 = datasetinfo.vae_preprocessor(X0)
        X0 = datasetinfo.vae.decode(X0).sample
    return X0

import math
def lmpdf(y):
    x1 = math.exp(-(math.log(y/(1-y))**2)/2)
    x2 = math.sqrt(2 * math.pi) * y * (1-y)
    return x1/x2

def train_rectified_flow(rank : int,
                         world_size : int,
                         model, 
                         optimizer, 
                         scheduler,
                         dataloader,
                         sampler, 
                         datasetinfo : ImageDatasetInfo,
                         use_classlabel = True,
                         max_timestep = 1000,
                         sampling_step = 50,
                         epoch = 16, 
                         dtype = torch.float16,
                         sample_type = "uniform"):
    if dtype == torch.float32:
        torch.set_float32_matmul_precision("high")
    device = rank
    loss_fun = torch.nn.MSELoss()
    max_class = len(datasetinfo.classes)
    def normalize_image(X):
        X = (X + 1)/2
        return X
    def eval_image():
        model.eval()
        assert world_size < 64 and 64 % world_size == 0
        total_batch_size = 64
        if datasetinfo.class_condition:
            class_labels = torch.arange(total_batch_size, device=rank) % max_class      
        else:
            class_labels = []
            for i in range(total_batch_size):
                class_labels.append(datasetinfo.dataset[i][1])
            class_labels = torch.stack(class_labels, dim=0).to(device)
        step = total_batch_size // world_size
        class_labels = class_labels[(rank*step):((rank+1)*step)]
        batch_size = step
        image_array = sample_image(rank,
                                   model, 
                                   max_timestep,
                                   sampling_step,
                                   batch_size = batch_size, 
                                   class_labels = class_labels,
                                   datasetinfo = datasetinfo,
                                   use_classlabel = use_classlabel,
                                   device= device)
        
        image_arrays = [torch.zeros_like(image_array, device=rank) for _ in range(world_size)]
        dist.all_gather(image_arrays, image_array)
        if rank == 0:
            image_array = torch.cat(image_arrays, dim = 0)
            if datasetinfo.is_latent:
                image_array = datasetinfo.vae_postprocessor.postprocess(image = image_array, output_type='pt')
            else:
                image_array = normalize_image(image_array)
            image_array = torchvision.utils.make_grid(image_array).permute((1,2,0)).detach().cpu().numpy()
            images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")
            wandb.log({"examples": images}, commit=False)

    timer = Timer()      
    scaler = torch.GradScaler()    

    def save_model(i):
        # only save the model in the main process
        if rank == 0:
            model_name = f"model-{i}.pt"
            # save the model here, first we need to unwrap the DDP module, then the torch.compile module
            torch.save(model.module._orig_mod, os.path.join(wandb.run.dir, model_name))
            # Save a model file manually from the current directory:
            wandb.save(model_name)
    for i in tqdm(range(epoch), position=0, leave=True):
        if i == 0:
            save_model("before")
        if i == 0:
            eval_image()
        # reset sampler
        if sampler is not None:
            sampler.set_epoch(i)
        if rank == 0:
            wandb.log({"epoch": i}, commit=False)
        for (image, class_labels) in tqdm(dataloader, position=0, leave=True):
            model.train()
            # image : shape (b, c, h, w)
            optimizer.zero_grad(set_to_none = True)
            image = image.to(device)
            class_labels = class_labels.to(device)
            noise = torch.randn(image.shape).to(device)
            assert image.dtype == noise.dtype
            if sample_type == "uniform":
                timestep = torch.rand(size=(image.size(0),)).to(device) * max_timestep
            elif sample_type == "lm":
                x = torch.tensor([lmpdf((i + 1/2)/max_timestep) + 0.1 for i in range(max_timestep)]).to(device)
                x = x/x.sum()
                timestep = torch.multinomial(x, image.size(0)).to(torch.float32) # [0, max_timestep-1]
                timestep += torch.rand(size=(image.size(0),)).to(device) # [0, max_timestep)
            elif sample_type == "lognorm0_1":
                x = torch.randn(size=(image.size(0),)).to(device)
                # map [-inf, inf] to [0, 1]
                x = torch.sigmoid(x)
                # scale the timestep into [0, max_timestep]
                timestep = x * max_timestep

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
            scheduler.step()
            dist.all_reduce(avg_values, op=dist.ReduceOp.AVG)
            if rank == 0:
                timer.step()
                wandb.log({"sample_per_second" : world_size * dataloader.batch_size * timer.rate()}, commit=False)
                wandb.log({"grad_scale" : scaler.get_scale()}, commit=False)
                wandb.log({"grad_norm": grad_norm.item()}, commit = False)
                wandb.log({"loss": avg_values[0].item()}, commit=True)
        
        save_model(i)
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
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # === reproducibility ===
    # we give a different seed on every model
    # this won't impact model weight, because DDP will copy weights
    # and this won't impact data sampler
    # we need to apply a different seed
    # otherwise, we always get the same seed on every device, which may break the diffusion model!
    import random
    random.seed(rank)
    import numpy as np
    np.random.seed(rank)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    # enable fp32 here
    torch.set_float32_matmul_precision("high")
    

def cleanup():
    dist.destroy_process_group()

# Define a closure here, so all the variables above are accessible here.... 
def main(rank, world_size, temp_dir, args, config):
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
    
    streaming.base.util.clean_stale_shared_memory()
    assert batch_size % world_size == 0
    per_device_batch_size = batch_size // world_size
    datasetinfo = get_dataset(rank, dataset_type, data_dir, temp_dir, batch_size=per_device_batch_size)
    print(datasetinfo)
    dataset = datasetinfo.dataset
    if datasetinfo.is_streaming_dataset:
        sampler = None
        # no need to shuffle here
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=per_device_batch_size, drop_last=True)
    else:
        if world_size == 1:
            sampler = None
            dataloader = make_dataloader(dataset, use_fake=use_fake, batch_size=batch_size, drop_last=True, shuffle=True)
        else:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed = 0, drop_last=True)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=per_device_batch_size, sampler=sampler, drop_last=True)

    # num_embeds_ada_norm = len(dataset.classes)
    model_config = config["model"]
    # hack : set condition type here with a parameter
    model_config["class_condition"] = datasetinfo.class_condition
    # we only call wandb in the first process
    sample_type = "uniform"
    if "sample_type" in config:
        sample_type = config["sample_type"]
    if rank == 0:
        print(model_config)
        wandb.login()
        mode = None
        if args.nowandb:
            mode = 'disabled'
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
                "config" : config,
                "args" : args,
                "sample_type" : sample_type
            },
            name = config["run"],
            mode = mode
        )
    model_type = "hf"
    if "model_type" in config:
        model_type = config["model_type"]
    assert model_type in ["hf", "origin"]
    if model_type == "hf":
        model = diffusers.DiTTransformer2DModel(**model_config)
    else:
        from model import DiTModelWrapper
        model = DiTModelWrapper(**model_config)
    model = torch.compile(model)
    model = model.to(rank)
    model = DDP(model)
    if rank == 0:
        print(model)
        print("model size:", sum([p.numel() for p in model.parameters()]))
    optimizer_type = "adamw"
    if "optimizer_type" in config:
        optimizer_type = config["optimizer_type"]
    assert optimizer_type in ["adamw", "lion"]
    if optimizer_type == "adamw":
        optimizer_class = torch.optim.AdamW
    elif optimizer_type == "lion":
        from lion_pytorch import Lion
        optimizer_class = Lion
    optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=optimizer_class, lr = lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=2000)

    train_rectified_flow(
                    rank,
                    world_size,
                    model, 
                    optimizer, 
                    scheduler,
                    dataloader, 
                    sampler,
                    datasetinfo,
                    max_timestep=max_timestep, 
                    epoch=epochs,
                    dtype = dtype,
                    sample_type=sample_type)
    if rank == 0:
        wandb.finish()
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str)
    parser.add_argument('--data_dir', type = str)
    parser.add_argument('--ddp', type=int)
    parser.add_argument('--nowandb', action='store_true')
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
    
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        mp.spawn(main, args=(world_size, temp_dir, args, config), nprocs=world_size, join=True)

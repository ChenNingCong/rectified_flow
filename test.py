import json
import copy
from os.path import join
root = "config/fashion-mnist-test"
with open(join(root, "test_base.json")) as f:
    base_config = json.load(f)
print(base_config)
def json_merge(base_config, newconfig):
    for i in newconfig:
        if i in base_config:
            if isinstance(newconfig[i], dict):
                assert isinstance(base_config[i], dict)
                json_merge(base_config[i], newconfig[i])
            else:
                base_config[i] = newconfig[i]
        else:
            base_config[i] = newconfig[i]
def _create_config(config_name : str, base_config, newconfig):
    base_config = copy.deepcopy(base_config)
    json_merge(base_config, newconfig)
    with open(join(root, config_name + ".json"), 'w') as f:
        f.write(json.dumps(base_config, indent=4))
    return base_config

def create_run_config(run_name : str, base_config, newconfig):
    newconfig["run"] = run_name
    return _create_config(run_name, base_config, newconfig)

print(create_run_config("fashion-mnist-fp16", base_config, {"dtype" : "fp16"}))
print(create_run_config("fashion-mnist-bf16", base_config, {"dtype" : "bf16"}))
print(create_run_config("fashion-mnist-fp32", base_config, {"dtype" : "fp32"}))

print(create_run_config("fashion-mnist-bf16-t-256", base_config, {"dtype" : "bf16", "max_timestep" : 256}))
print(create_run_config("fashion-mnist-bf16-t-512", base_config, {"dtype" : "bf16", "max_timestep" : 512}))
print(create_run_config("fashion-mnist-bf16-t-1024", base_config, {"dtype" : "bf16", "max_timestep" : 1024}))

print(create_run_config("fashion-mnist-bf16-patch-4", base_config, {"dtype" : "bf16", "model" : {"patch_size" : 4}}))
print(create_run_config("fashion-mnist-bf16-patch-7", base_config, {"dtype" : "bf16", "model" : {"patch_size" : 7}}))


print(create_run_config("cifar-bf16-512epoch", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "cifar-10", 
                                                             "epochs" : 512,
                                                             "model" : {"in_channels" : 3, "sample_size": 32, "in_channels" : 3}}))

print(create_run_config("cifar-bf16-512epoch-10M", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "cifar-10", 
                                                             "epochs" : 512,
                                                             "model" : {
                                                                 "num_attention_heads": 16,
                                                                 "attention_head_dim": 32,
                                                                 "sample_size": 32,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 3}}))

print(create_run_config("cifar-bf16-512epoch-20M", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "cifar-10", 
                                                             "epochs" : 512,
                                                             "model" : {
                                                                 "num_attention_heads": 16,
                                                                 "attention_head_dim": 32,
                                                                 "sample_size": 32,
                                                                 "norm_num_groups": 32,
                                                                 "num_layers": 8,
                                                                 "in_channels" : 3}}))

print(create_run_config("cifar-bf16-512epoch-50M", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "cifar-10", 
                                                             "epochs" : 512,
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 32,
                                                                 "sample_size": 32,
                                                                 "num_layers": 12,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 3}}))

print(create_run_config("cifar-bf16-512epoch-50M-t-512", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "cifar-10", 
                                                             "epochs" : 512,
                                                             "max_timestep" : 512,
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 32,
                                                                 "sample_size": 32,
                                                                 "num_layers": 12,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 3}}))

print(create_run_config("cifar-bf16-512epoch-50M-t-1024", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "cifar-10", 
                                                             "epochs" : 512,
                                                             "max_timestep" : 1024,
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 32,
                                                                 "sample_size": 32,
                                                                 "num_layers": 12,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 3}}))

# imagenet
# imagenet is 10x larger than cifar, so we shrink the epoch size...
print(create_run_config("imagenet32-bf16-128epoch-50M-t-256", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet-32", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 256,
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 32,
                                                                 "sample_size": 32,
                                                                 "num_layers": 12,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 3}})) 

# imagenet
# imagenet is 10x larger than cifar, so we shrink the epoch size...
print(create_run_config("imagenet32-fp32-128epoch-50M-t-256", base_config, {"dtype" : "fp32", 
                                                             "dataset_type" : "imagenet-32", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 512,
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 32,
                                                                 "sample_size": 32,
                                                                 "num_layers": 12,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 3}})) 


# imagenet is 10x larger than cifar, so we shrink the epoch size...
print(create_run_config("imagenet32-bf16-128epoch-100M-t-256", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet-32", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 256,
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 32,
                                                                 "sample_size": 32,
                                                                 "num_layers": 24,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 3}}))                                                                   


# imagenet is 10x larger than cifar, so we shrink the epoch size...
print(create_run_config("imagenet.uint8-bf16-128epoch-100M-t-256", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 256,
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 32,
                                                                 "sample_size": 32,
                                                                 "num_layers": 24,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}}))   

# imagenet is 10x larger than cifar, so we shrink the epoch size...
print(create_run_config("imagenet.uint8-bf16-128epoch-200M-t-256", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 256,
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 24,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}}))   


# imagenet is 10x larger than cifar, so we shrink the epoch size...
print(create_run_config("imagenet.uint8-origin-bf16-128epoch-50M-t-256", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 256,
                                                             "model_type" : "origin",
                                                             "model" : {
                                                                 "num_attention_heads": 6,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 12,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}}))   

print(create_run_config("imagenet.uint8-origin-bf16-128epoch-100M-t-256", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 256,
                                                             "model_type" : "origin",
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 12,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}}))   
                                                                

print(create_run_config("imagenet.uint8-origin-bf16-128epoch-100M-t-1000", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 1000,
                                                             "model_type" : "origin",
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 12,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}}))                                                                 


print(create_run_config("imagenet.uint8-origin-bf16-128epoch-200M-t-1000", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 1000,
                                                             "model_type" : "origin",
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 24,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}}))                                                                   


# test non-uniform sampling

print(create_run_config("fashion-mnist-bf16-lmsample", base_config, {"dtype" : "bf16", "sample_type": "lm"}))     
print(create_run_config("imagenet.uint8-origin-bf16-128epoch-100M-t-1000-lmsample", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 1000,
                                                             "model_type" : "origin",
                                                             "sample_type": "lm",
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 12,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}}))                                               

print(create_run_config("imagenet.uint8-origin-fp16-128epoch-100M-t-1000-lmsample", base_config, {"dtype" : "fp16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 1000,
                                                             "model_type" : "origin",
                                                             "sample_type": "lm",
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 12,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}}))                                               
  
print(create_run_config("imagenet.uint8-origin-fp32-128epoch-100M-t-1000-lmsample", base_config, {"dtype" : "fp32", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 1000,
                                                             "model_type" : "origin",
                                                             "sample_type": "lm",
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 12,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}}))  

print(create_run_config("imagenet.uint8-origin-bf16-128epoch-200M-t-1000-lmsample", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 1000,
                                                             "model_type" : "origin",
                                                             "sample_type": "lm",
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 24,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}})) 
         

print(create_run_config("imagenet.uint8-origin-bf16-128epoch-200M-t-1000-lmsample-lion", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 1000,
                                                             "model_type" : "origin",
                                                             "sample_type": "lm",
                                                             "lr" : 1e-5,
                                                             "optimizer_type" : "lion",
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 24,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}})) 

print(create_run_config("imagenet.uint8-origin-bf16-128epoch-200M-t-1000-lmsample-rotary", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 1000,
                                                             "model_type" : "origin",
                                                             "sample_type": "lm",
                                                             "model" : {
                                                                 "attention_type" : "RotaryAttention",
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 24,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}})) 

print(create_run_config("mj-origin-bf16-128epoch-100M-t-1000-lmsample", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "mj", 
                                                             "batch_size" : 512,
                                                             "epochs" : 128,
                                                             "max_timestep" : 1000,
                                                             "model_type" : "origin",
                                                             "sample_type": "lm",
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 12,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}})) 
                                                                                                                 

print(create_run_config("imagenet.uint8-origin-bf16-128epoch-200M-t-512-lmsample", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 512,
                                                             "model_type" : "origin",
                                                             "sample_type": "lm",
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 24,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}}))                                                                                                             


print(create_run_config("mj-origin-bf16-128epoch-200M-t-512-lmsample", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "mj", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 512,
                                                             "model_type" : "origin",
                                                             "sample_type": "lm",
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 24,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}})) 

print(create_run_config("imagenet.uint8-origin-bf16-128epoch-200M-t-512-lognormsample", base_config, {"dtype" : "bf16", 
                                                             "dataset_type" : "imagenet.uint8", 
                                                             "epochs" : 128,
                                                             "max_timestep" : 512,
                                                             "model_type" : "origin",
                                                             "sample_type": "lognorm0_1",
                                                             "model" : {
                                                                 "num_attention_heads": 12,
                                                                 "attention_head_dim": 64,
                                                                 "sample_size": 32,
                                                                 "num_layers": 24,
                                                                 "num_embeds_ada_norm": 1000,
                                                                 "norm_num_groups": 32,
                                                                 "in_channels" : 4}}))   
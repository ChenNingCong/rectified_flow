srun -N 1 -c 8  -n 1 --mem=16384 --time=1440 --partition=academic --gres=gpu:1 python /home/nchen3/autollm/rectified_flow/main.py --config /home/nchen3/autollm/rectified_flow/config/fashion-mnist-test/imagenet.uint8-bf16-128epoch-100M-t-256.json --data_dir /home/nchen3/autollm/data/ --ddp 4


python /home/nchen3/autollm/rectified_flow/main.py --config /home/nchen3/autollm/rectified_flow/config/fashion-mnist-test/imagenet.uint8-origin-fp32-128epoch-100M-t-1000-lmsample.json --data_dir /home/nchen3/autollm/data/ --ddp 4
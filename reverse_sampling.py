import os
import torch
import torch.distributed as dist
from transformers import GPT2TokenizerFast
from load_model import load_model
import utils
import graph_lib
import noise_lib
from model import SEDD
from model.ema import ExponentialMovingAverage
from torch.nn.parallel import DistributedDataParallel as DDP
from itertools import chain
import losses
import torch.nn.functional as F

def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_samples(file_path):
    """Load and split samples from the output file."""
    with open(file_path, 'r') as f:
        text = f.read()
    samples = text.split('<|endoftext|>')
    samples = [s.strip() for s in samples if s.strip()]
    return samples

def forward_diffusion(model, graph, noise, x_0, steps=1024, device='cuda'):
    """Apply forward diffusion process to recover noise."""
    timesteps = torch.linspace(0, 1, steps + 1, device=device)[:-1]
    dt = 1 / steps
    
    x = x_0
    for i in range(steps):
        t = timesteps[i] * torch.ones(x.shape[0], device=device)
        sigma, dsigma = noise(t)
        score = model(x, sigma)
        
        forward_rate = -dt * dsigma[..., None] * graph.reverse_rate(x, score)
        x = graph.sample_rate(x, forward_rate)
    
    return x

def main():
    # Initialize distributed setup
    rank = 0
    world_size = 1
    port = 29500
    setup(rank, world_size, port)
    
    # Match your specific configuration
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    
    # Load the same configuration you used for sampling
    root_dir = '/home/avbagchi_umass_edu/Score-Entropy-Discrete-Diffusion/configs'
    cfg = utils.load_hydra_config_from_run(root_dir, True)
    
    # Load model components using the same checkpoint
    checkpoint_meta_dir = '/scratch3/workspace/avbagchi_umass_edu-data_center/uniform/checkpoints/checkpoint_9.pth'
    
    # Initialize model components
    graph = graph_lib.get_graph(cfg, device)
    score_model = SEDD(cfg).to(device)
    score_model = DDP(score_model, device_ids=[0], static_graph=True)
    
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)
    
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[0], static_graph=True)
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    scaler = torch.cuda.amp.GradScaler()
    
    # Load state
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0)
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    # Load samples
    samples = load_samples('output2.txt')
    
    # Process each sample
    batch_size = cfg.training.batch_size // cfg.ngpus  # Match your sampling batch size
    sampling_eps = 1e-5  # Match your sampling epsilon
    
    for idx, sample in enumerate(samples):
        # Tokenize the text
        tokens = tokenizer(sample, return_tensors='pt').input_ids.to(device)
        
        # Ensure the sequence length is 1024 (cfg.model.length)
        if tokens.shape[1] < cfg.model.length:
            tokens = F.pad(tokens, (0, cfg.model.length - tokens.shape[1]), value=tokenizer.pad_token_id)
        elif tokens.shape[1] > cfg.model.length:
            tokens = tokens[:, :cfg.model.length]
        
        # Apply EMA weights as done in sampling
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        
        # Apply forward diffusion
        recovered_noise = forward_diffusion(
            score_model, 
            graph, 
            noise, 
            tokens, 
            steps=1024,  # Match your sampling steps
            device=device
        )
        
        # Restore original weights
        ema.restore(score_model.parameters())
        
        # Save the recovered noise
        torch.save(recovered_noise, f'recovered_noise_{idx}.pt')
        # print(f"Processed sample {idx + 1}/{len(samples)}")
        print(recovered_noise)

    cleanup()  # Add cleanup at the end

if __name__ == "__main__":
    main()
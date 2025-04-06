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
    """Load single sample from file."""
    with open(file_path, 'r') as f:
        text = f.read()
    print(f"Loaded text (first 200 chars):\n{text[:200]}...")
    return [text]  # Return as single-item list to maintain compatibility

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

def calculate_green_matches(recovered_tokens):
    vocab_size = 50257 
    sequence_length = recovered_tokens.shape[1]  
    matches = 0
    total = sequence_length  
    
    for pos in range(sequence_length):
        torch.manual_seed(pos)
        pos_green_mask = torch.randint(0, 2, (vocab_size,), device=recovered_tokens.device)
        
        token = recovered_tokens[0, pos]  # [0] because batch size is 1
        
        if pos_green_mask[token] == 1:
            matches += 1
    
    percent_match = (matches / total) * 100
    return percent_match

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
    
    # Load single sample
    samples = load_samples('output.txt')
    breakpoint()
    print(f"\nProcessing single sample")
    
    # Process the sample
    sample = samples[0]  # Get the single sample
    
    # Tokenize the text
    tokens = tokenizer(sample, return_tensors='pt').input_ids.to(device)
    print(f"Token shape before padding: {tokens.shape}")
    breakpoint()
    # Ensure the sequence length is 1024
    if tokens.shape[1] < cfg.model.length:
        tokens = F.pad(tokens, (0, cfg.model.length - tokens.shape[1]), value=tokenizer.pad_token_id)
    elif tokens.shape[1] > cfg.model.length:
        tokens = tokens[:, :cfg.model.length]
    print(f"Token shape after padding: {tokens.shape}")
    # breakpoint()
    # Apply EMA weights
    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    
    # Apply forward diffusion
    # print("\nStarting forward diffusion...")
    # recovered_noise = forward_diffusion(
    #     score_model, 
    #     graph, 
    #     noise, 
    #     tokens, 
    #     steps=1024,
    #     device=device
    # )
    sample = torch.load('sample.pt')
    print(sample.shape)
    print("\nAnalyzing green list matches...")
    percent_green_matches = calculate_green_matches(sample)
    print(f"{percent_green_matches:.2f}% of tokens are in their position-specific green list")
    
    
    # Restore original weights
    ema.restore(score_model.parameters())

    # torch.save(recovered_noise, 'recovered_noise_3.pt')
    

    cleanup()

if __name__ == "__main__":
    main()
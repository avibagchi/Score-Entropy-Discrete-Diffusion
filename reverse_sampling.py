import torch
import torch.nn.functional as F
from model import utils as mutils
import utils
from model import SEDD
import noise_lib
import graph_lib
from torch.nn.parallel import DistributedDataParallel as DDP

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
root_dir = '/home/avbagchi_umass_edu/Score-Entropy-Discrete-Diffusion/configs'
cfg = utils.load_hydra_config_from_run(root_dir, True)

# Setup model and load checkpoint
graph = graph_lib.get_graph(cfg, device)
score_model = SEDD(cfg).to(device)
score_model = DDP(score_model, device_ids=[0], static_graph=True)

# Setup noise
noise = noise_lib.get_noise(cfg).to(device)
noise = DDP(noise, device_ids=[0], static_graph=True)

# Load checkpoint
checkpoint_meta_dir = '/scratch3/workspace/avbagchi_umass_edu-data_center/uniform/checkpoints/checkpoint_9.pth'
state = dict(model=score_model, noise=noise, step=0)
state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)

@torch.no_grad()
def reverse_deterministic_sampling(model, final_tokens, graph, noise, steps=128, eps=1e-5, device='cuda'):
    sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
    
    # Start with your output token matrix [64, 1024]
    x = final_tokens.to(device)
    
    # Create timesteps in reverse (from eps to 1 instead of 1 to eps)
    timesteps = torch.linspace(eps, 1, steps + 1, device=device)
    dt = (1 - eps) / steps

    # First undo the denoising step
    t = timesteps[0] * torch.ones(x.shape[0], 1, device=device)
    sigma = noise.total_noise(t)
    score = sampling_score_fn(x, sigma)
    stag_score = graph.staggered_score(score, sigma)
    probs = stag_score * graph.transp_transition(x, sigma)
    if graph.absorb:
        probs = probs[..., :-1]  # Truncate probabilities like in original denoiser
    # Convert probabilities back to pre-denoised state
    x = torch.argmax(probs, dim=-1)  # Get most likely pre-denoised state

    # Run through timesteps in reverse
    for i in range(steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
        sigma, dsigma = noise.total_noise(t), noise.rate_noise(t)
        
        # Get score and compute reverse transition
        score = sampling_score_fn(x, sigma)
        rev_rate = dt * dsigma[..., None] * graph.reverse_rate(x, score)
        
        # Apply reverse transition
        probs = F.one_hot(x, num_classes=graph.dim).to(rev_rate) + rev_rate
        x = torch.argmax(probs, dim=-1)  # Get most likely previous state
    
    return x

# Use the objects
output_tokens = torch.tensor(...).shape([64, 1024])  # Your token matrix
initial_noise = reverse_deterministic_sampling(
    model=score_model,
    final_tokens=output_tokens,
    graph=graph,
    noise=noise,
    device=device
)
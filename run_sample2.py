import os
import torch
import torch.distributed as dist
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.nn.functional as F

import utils
import sampling
from model import SEDD
from model.ema import ExponentialMovingAverage
import losses
from itertools import chain
import noise_lib
import graph_lib
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def sample(rank, world_size, cfg, port):
    setup(rank, world_size, port)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(rank)

    # work_dir = cfg.data.cache_dir
    # sample_dir = os.path.join(work_dir, "samples")
    # checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")

    checkpoint_meta_dir = '/scratch3/workspace/avbagchi_umass_edu-data_center/uniform/checkpoints/checkpoint_9.pth'
    # checkpoint_meta_dir = '/scratch3/workspace/avbagchi_umass_edu-data_center/absorb/checkpoints/checkpoint_9.pth'

    # if rank == 0:
    #     utils.makedirs(sample_dir)

    graph = graph_lib.get_graph(cfg, device)
    score_model = SEDD(cfg).to(device)
    score_model = DDP(score_model, device_ids=[rank], static_graph=True)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    scaler = torch.cuda.amp.GradScaler()

    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0)
    print(state)
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    sampling_eps = 1e-5
    sampling_shape = (cfg.training.batch_size // cfg.ngpus, cfg.model.length)
    sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    step = state['step']
    print(f"Generating samples at step: {step}")

    # this_sample_dir = os.path.join(sample_dir, f"iter_{step}")
    # utils.makedirs(this_sample_dir)

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    sample = sampling_fn(score_model)
    ema.restore(score_model.parameters())

    sentences = tokenizer.batch_decode(sample)
    for i, sentence in enumerate(sentences):
        print(f"Sample {i}:\n{sentence}\n{'='*80}")

    # file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
    # with open(file_name, 'w') as file:
    #     for sentence in sentences:
    #         file.write(sentence + "\n")
    #         file.write("="*80 + "\n")

    # Evaluate perplexity (if enabled)
    if cfg.eval.perplexity:
        with torch.no_grad():
            eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
            total_perplexity = 0
            batch_size = cfg.eval.perplexity_batch_size
            num_batches = sample.shape[0] // batch_size

            for i in range(num_batches):
                s = sample[i * batch_size:(i + 1) * batch_size]
                loss, logits = eval_model(s, labels=s)[:2]
                logits = logits.transpose(-1, -2)
                perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                total_perplexity += perplexity

            total_perplexity /= num_batches
            dist.all_reduce(total_perplexity)
            total_perplexity /= world_size
            print(f"Generative Perplexity at step {step}: {total_perplexity:.3f}")

    cleanup()


if __name__ == "__main__":
    root_dir = '/home/avbagchi_umass_edu/Score-Entropy-Discrete-Diffusion/configs'
    cfg = utils.load_hydra_config_from_run(root_dir, True)
    sample(rank=0, world_size=1, cfg=cfg, port=29500)

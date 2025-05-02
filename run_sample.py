import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.distributed as dist
import os
def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

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

def calculate_green_matches_no_index(recovered_tokens):
    vocab_size = 50257 
    sequence_length = recovered_tokens.shape[1]
    max_match_percent = 0
    best_start = 0
    
    n = 5
    match_arr = []
    for start in range(0, n): 
        matches = 0        
        for pos in range(sequence_length):
            torch.manual_seed((pos+start) % n) 
            pos_green_mask = torch.randint(0, 2, (vocab_size,), device=recovered_tokens.device)
            
            token = recovered_tokens[0, pos]  
            
            if pos_green_mask[token] == 1:
                matches += 1
        
        percent_match = (matches / sequence_length) * 100
        match_arr.append([start, percent_match])
        if percent_match > max_match_percent:
            max_match_percent = percent_match
            best_start = start
    
    return max_match_percent, best_start, match_arr


def main():
    
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()
    setup(0, 1, 29500)

    
    device = torch.device('cuda')
    model, graph, noise = load_model(args.model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Print token at specific index
    # 2000 = mind
    target_token_idx = 2000  # Change this to the index you want to amplify
    print(f"Token at index {target_token_idx}: {tokenizer.decode([target_token_idx])}")
    # breakpoint()

    # added first amplification argument
    amplification = 5
    sampling_fn = sampling.get_pc_sampler(amplification,
        graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device
    )

    samples = sampling_fn(model)

    text_samples = tokenizer.batch_decode(samples)
    for i in text_samples:
        print(i)
        print("=================================================")
    
    
    with torch.no_grad():
        eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
        total_perplexity = 0
        batch_size = 1
        num_batches = samples.shape[0] // batch_size

        for i in range(num_batches):
            s = samples[i * batch_size:(i + 1) * batch_size]
            loss, logits = eval_model(s, labels=s)[:2]
            logits = logits.transpose(-1, -2)
            perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
            total_perplexity += perplexity

        total_perplexity /= num_batches
        dist.all_reduce(total_perplexity)
        print(f"Generative Perplexity at step: {total_perplexity:.3f}")
    
    # breakpoint()
    max_match_percent, best_start, match_arr = calculate_green_matches_no_index(samples)
    print(f"Percent match: {max_match_percent}")
    cleanup()

if __name__=="__main__":
    main()
import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import sampling
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.distributed as dist
import os
import math
import csv

def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_green_token_positions(samples, gamma, vocab_size=50257):
    """Get positions of green tokens in the generated samples"""
    sequence_length = samples.shape[1]
    green_positions = []
    
    n = 5
    for start in range(n):
        positions = []
        for pos in range(sequence_length):
            # Stop when we reach the EOS token (50256)
            if samples[0, pos] == 50256:
                break
                
            torch.manual_seed((pos+start) % n) 
            # Create exactly gamma*|V| green tokens and (1-gamma)*|V| red tokens
            num_green = int(gamma * vocab_size)
            pos_green_mask = torch.zeros(vocab_size, device=samples.device)
            pos_green_mask[:num_green] = 1
            pos_green_mask = pos_green_mask[torch.randperm(vocab_size, device=samples.device)]
            
            token = samples[0, pos]  
            
            if pos_green_mask[token] == 1:
                positions.append(pos)
        
        green_positions.append(positions)
    
    # Find the start that gives maximum green tokens
    max_green_count = 0
    best_start = 0
    for start, positions in enumerate(green_positions):
        if len(positions) > max_green_count:
            max_green_count = len(positions)
            best_start = start
    
    return green_positions[best_start], best_start

def format_text_with_green_tokens(text, green_positions, tokenizer):
    """Format text with green tokens in bold"""
    tokens = tokenizer.encode(text)
    formatted_parts = []
    
    for i, token in enumerate(tokens):
        if i in green_positions:
            # Use markdown bold formatting
            token_text = tokenizer.decode([token])
            formatted_parts.append(f"**{token_text}**")
        else:
            token_text = tokenizer.decode([token])
            formatted_parts.append(token_text)
    
    return ''.join(formatted_parts)

def calculate_green_matches_no_index(recovered_tokens, gamma=0.5):
    vocab_size = 50257 
    sequence_length = recovered_tokens.shape[1]
    max_match_percent = 0
    best_start = 0
    actual_length_used = 0  # Initialize this variable
    
    n = 5
    match_arr = []
    max_num_matches = 0
    for start in range(0, n): 
        matches = 0
        actual_length = 0  # Track actual sequence length until EOS
        
        for pos in range(sequence_length):
            # Stop when we reach the EOS token (50256)
            if recovered_tokens[0, pos] == 50256:
                break
                
            torch.manual_seed((pos+start) % n) 
            # Create exactly gamma*|V| green tokens and (1-gamma)*|V| red tokens
            num_green = int(gamma * vocab_size)
            pos_green_mask = torch.zeros(vocab_size, device=recovered_tokens.device)
            pos_green_mask[:num_green] = 1
            pos_green_mask = pos_green_mask[torch.randperm(vocab_size, device=recovered_tokens.device)]
            
            token = recovered_tokens[0, pos]  
            
            if pos_green_mask[token] == 1:
                matches += 1
            
            actual_length += 1
        
        # Use actual_length instead of sequence_length for percentage calculation
        if actual_length > 0:
            percent_match = matches / actual_length
        else:
            percent_match = 0
            
        match_arr.append([start, percent_match])
        if percent_match > max_match_percent:
            max_match_percent = percent_match
            actual_length_used = actual_length
            max_num_matches = matches
            best_start = start
    
    return max_match_percent, actual_length_used, max_num_matches, best_start

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--prefix", type=str, default="The Federal Reserve is ")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    prefix_ids = tokenizer(args.prefix).input_ids
    suffix_ids = tokenizer(args.suffix).input_ids
    input_ids = prefix_ids + suffix_ids
    input_locs = list(range(len(prefix_ids))) + list(range(1024-len(suffix_ids), 1024))

    # more generaly commands can be defined with something like below:
    # input_ids = [0, 1, 512, 8080, 50256, 20000]
    # input_locs = [5, 6, 19, 20, 1000, 10001]

    input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(args.batch_size, 1)

    def proj_fun(x):
        x[:, input_locs] = input_ids
        return x
    
    device = torch.device('cuda')
    model, graph, noise = load_model(args.model_path, device)
    
    # Watermarking parameters
    vocab_size = 50258
    sequence_length = 1024
    gamma_list = [0.025]  # [0.9, 0.75, 0.5, 0.25, 0.1]
    amplification_arr = [1]  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000, 5000, 10000]
    is_tree_ring = True
    
    for gamma in gamma_list:
        # Generate green masks for this gamma value
        green_masks = []
        for pos in range(sequence_length):
            n = 5
            torch.manual_seed(pos % n)  # Seed based on position
            # Create exactly gamma*|V| green tokens and (1-gamma)*|V| red tokens
            num_green = int(gamma * vocab_size)
            pos_green_mask = torch.zeros(vocab_size, device=device)
            pos_green_mask[:num_green] = 1
            pos_green_mask = pos_green_mask[torch.randperm(vocab_size, device=device)]
            green_masks.append(pos_green_mask)
        
        green_mask = torch.stack(green_masks, dim=0)
        green_mask = green_mask.unsqueeze(0)

        water_data_arr = []
    
        for amplification in amplification_arr:
            # Create a separate file for each gamma and amplification combination
            for step_to_watermark in range(200, 201, 50):  # changed from 0, 1025, 50
                for model_seed in range(1, 2, 1):
                    filename = f'table4/seed_{model_seed}_gamma_{gamma}_amp_{amplification}.csv'
                    setup(0, 1, 29500)
                    torch.manual_seed(model_seed)
                    sampling_fn = sampling.get_pc_sampler(is_tree_ring, amplification, green_mask, step_to_watermark,
                        graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device, proj_fun=proj_fun
                    )
                    
                    samples = proj_fun(sampling_fn(model))

                    # Get green token positions
                    green_positions, best_start = get_green_token_positions(samples, gamma)
                    
                    text_samples = tokenizer.batch_decode(samples)
                    for i, text in enumerate(text_samples):
                        # Format text with green tokens in bold
                        formatted_text = format_text_with_green_tokens(text, green_positions, tokenizer)
                        print(formatted_text)
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

                    cleanup()
                
                    # Calculate watermark metrics
                    max_match_percent, actual_length_used, max_num_matches, best_start = calculate_green_matches_no_index(samples, gamma)
                    true_num_green = gamma * actual_length_used
                    if math.sqrt(true_num_green * (1-gamma)) == 0:
                        z_score = 0
                    else:
                        z_score = (max_num_matches - true_num_green) / math.sqrt(true_num_green * (1-gamma))
                    print(f"Percent match: {max_match_percent}")
                    
                    # Get the actual green tokens
                    green_tokens = []
                    tokens = tokenizer.encode(text_samples[0])  # Get tokens from the first sample
                    for pos in green_positions:
                        if pos < len(tokens):
                            green_tokens.append(tokenizer.decode([tokens[pos]]))
                    
                    water_data = {
                        "model_seed": model_seed,
                        "gamma": gamma,
                        "amplification": amplification,
                        "step_to_watermark": step_to_watermark,
                        "max_match_percent": max_match_percent,
                        "perplexity": float(total_perplexity.item()),
                        "z_score": float(z_score),
                        "actual_length_used": actual_length_used,
                        "max_num_matches": max_num_matches,
                        "best_start": best_start,
                        "green_tokens": green_tokens,
                        "green_token_positions": green_positions
                    }
                    print(f"Water data: {water_data}")
                    water_data_arr.append(water_data)
                
                # Save to CSV file after going through all seeds
                # with open(filename, 'w', newline='') as csvfile:
                #     writer = csv.DictWriter(csvfile, fieldnames=water_data.keys())
                #     writer.writeheader()
                #     writer.writerows(water_data_arr)
                print(f"Saved results to {filename}")
                
                # Reset water_data_arr for next amplification
                water_data_arr = []

if __name__=="__main__":
    main()
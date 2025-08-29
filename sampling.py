import abc
import torch
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


# change this
@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, amplification=0):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)
        
        # Ensure score has the correct dimension
        # if score.shape[-1] != self.graph.dim:
        #     score = score[..., :self.graph.dim]
            
        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(amplification, x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, green_mask, amplification=0):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        breakpoint()
        score = score_fn(x, curr_sigma)

        if (amplification > 0):
            green_mask = green_mask.to(score.device)
            # New formula: amplification=1 does nothing, amplification=2 scales by 2
            score = score * (1 + green_mask * (amplification - 1))
            
        # breakpoint()
        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)

        return sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        # breakpoint()
        return sample_categorical(probs) # random choice from probs, need to change this
                       

def get_sampling_fn(amplification, config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(amplification=amplification, graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn

def embed_tree_ring(x, batch_dims, device, mask_radius=4):
    """Create a watermarked noise vector by modifying the Fourier space."""
    shape = (*batch_dims,)  # e.g., (batch, seq_len)
    #x = torch.randn(*shape, device=device)  # Gaussian noise

    # FFT over last dimension (sequence)
    x_fft = torch.fft.fft(x, dim=-1)

    # Build circular low-frequency mask M
    seq_len = shape[-1]
    freqs = torch.fft.fftfreq(seq_len, d=1.0).to(device)
    mask = torch.abs(freqs) < (mask_radius / seq_len)

    # Create watermark pattern in Fourier space
    watermark = torch.randn_like(x_fft)
    x_fft[..., mask] = watermark[..., mask]  # Insert watermark pattern

    # Inverse FFT to real domain
    x_watermarked = torch.fft.ifft(x_fft, dim=-1).real
    # breakpoint()
    return x_watermarked
    

def get_pc_sampler(is_tree_ring, amplification, green_mask, step_to_watermark, graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        
       
        # inital noise vector watermark (does not work)
        # watermark = False 

        # if watermark:
        #     import prc
        #     n = batch_dims[0] * batch_dims[1]  # change this? 
        #     message = prc.str_to_bin("W") # change to one bit  
        #     encoding_key, _ = prc.KeyGen(n=n)   
        #     encoded_watermark = prc.Encode(encoding_key, message)  
        #     encoded_watermark = encoded_watermark.to(device)
        #     x = torch.clamp(encoded_watermark.reshape(*batch_dims).to(device).long(), 0, 50256)  # Changed to match vocab size
        #     x = torch.full(encoded_watermark.reshape(*batch_dims).shape, 3000, device=device, dtype=torch.long)
        # else:
        
        if is_tree_ring:
            x = graph.sample_limit(*batch_dims).to(device)
            # breakpoint()
            x = embed_tree_ring(x, batch_dims, device)  # shape: [B, L]
            x = torch.clamp(x.round().long(), 0, 50257)
            # breakpoint()
        else:
            x = graph.sample_limit(*batch_dims).to(device)
        
        # end initial noise vector watermark
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps
        
        for i in range(steps):
            #breakpoint()
            # if i == 1023:
            #     breakpoint()     
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            if i <= step_to_watermark: # changed here
                current_amplification = amplification 
            else:
                current_amplification = 0
            # breakpoint()
            x = predictor.update_fn(sampling_score_fn, x, t, dt, green_mask, current_amplification)
            
        # breakpoint()
        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
        # breakpoint()
        return x
    
    return pc_sampler


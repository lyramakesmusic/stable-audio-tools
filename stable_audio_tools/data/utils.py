import math
import random
import torch

from torch import nn
from typing import Tuple

class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

class PadCrop_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int, torch.Tensor]:
        n_channels, n_samples = source.shape
    
        # Convert total samples to total duration in seconds
        total_duration_seconds = math.ceil(n_samples / self.sample_rate)
    
        # Calculate the maximum offset in samples to ensure the chunk fits
        max_offset_in_samples = n_samples - self.n_samples
        max_offset_in_samples = max(0, max_offset_in_samples)
    
        # If randomize is True, choose a random starting point, otherwise start at 0
        if self.randomize and max_offset_in_samples > 0:
            offset = random.randint(0, max_offset_in_samples)
        else:
            offset = 0
    
        # Calculate the start time of the chunk in seconds
        seconds_start = offset / self.sample_rate
    
        # Calculate the start and end times of the chunk
        t_start = offset / (max_offset_in_samples + self.n_samples)
        t_end = (offset + self.n_samples) / (max_offset_in_samples + self.n_samples)
    
        # Create the chunk
        chunk = source.new_zeros([n_channels, self.n_samples])
    
        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]
        
        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
    
        return chunk, t_start, t_end, seconds_start, total_duration_seconds, padding_mask

class PhaseFlipper(nn.Module):
    "Randomly invert the phase of a signal"
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        return -signal if (random.random() < self.p) else signal
        
class Mono(nn.Module):
  def __call__(self, signal):
    return torch.mean(signal, dim=0, keepdims=True) if len(signal.shape) > 1 else signal

class Stereo(nn.Module):
  def __call__(self, signal):
    signal_shape = signal.shape
    # Check if it's mono
    if len(signal_shape) == 1: # s -> 2, s
        signal = signal.unsqueeze(0).repeat(2, 1)
    elif len(signal_shape) == 2:
        if signal_shape[0] == 1: #1, s -> 2, s
            signal = signal.repeat(2, 1)
        elif signal_shape[0] > 2: #?, s -> 2,s
            signal = signal[:2, :]    

    return signal

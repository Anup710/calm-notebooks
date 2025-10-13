# In this file we will replicate the gpt2 architecture (decoder only) with the same hyperparams
# openai has used, then we load the pretrained weight from hugging face in the GPT class. 
# finally we generate from the model, just to verify compatibility of our architecture 
# with the openai gpt2 one. 

import math 
from dataclasses import dataclass
import torch
import torch.nn as nn 
from torch.nn import functional as f


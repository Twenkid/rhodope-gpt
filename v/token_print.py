#Print the tokens: for cyrillic
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
"""
#import time
import datetime
import sys

import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
#from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Config, GPT2Tokenizer
#from transformers import  GPT2Tokenizer

from pathlib import Path

#23-5-2023
#Thanks: 
# https://www.google.com/search?q=gpt2+tokenize+vocab+encoding&oq=gpt2+tokenize+vocab+encoding&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiiBDIHCAIQABiiBDIHCAMQABiiBDIHCAQQABiiBDIHCAUQABiiBNIBCTEwMDAyajBqN6gCALACAA&sourceid=chrome&ie=UTF-8
# https://discuss.huggingface.co/t/which-encoding-does-gpt2-vocabulary-file-use/8875/3
#f = open("Z:\\vocab1.json", "rt")

tokenizer_path = "z:/tokenized_data/" 
print(tokenizer_path)
tokenizerLOAD = GPT2Tokenizer(vocab_file=tokenizer_path+"vocab.json", merges_file=tokenizer_path+'merges.txt')

bg = []
for token in tokenizerLOAD.get_vocab().keys():
  bg.append(tokenizerLOAD.convert_tokens_to_string(token))

print(bg)







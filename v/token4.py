#DONT RUN IF NOT ALREADY SAVED TOKENIZER ETC.!
import torch
import torch.nn as nn
from torch.nn import functional as F
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

# A MIXTURE OF THE BASIC TRANSFORMER BY KARPATHY and
# Todor's GPT2-Medium Training code to use the GPT2 tokenizers
# from huggingface transformers
# 20-5-2023

# SET in another cell for easy access
#load_tokeinzer = True #False first time for training it - training is slow, 9-10 min for 144 MB
#will_train = True #False #skip encoding the Dataset etc.

#model_path = "z:\\nn\\cap12_256-16-16-block128-batch-32_Vidal.nn"
#model_path_backup = "z:\\nn\\cap12_256-16-16-block128-batch-32_Vidal_BACKUP.nn"


LOAD_TO_TRAIN = "2" #a string!
LOAD_TO_GEN = "1"
TRAIN_NEW = "3"
load_tokeinzer = True #False #TRAIN NEW True #False first time for training it - training is slow, 9-10 min for 144 MB

i = input("WILL TRAIN? 1/y?")
will_train = i=="1" or i=="y" 
print("will_train = ", str(will_train))
#will_train = True # False #True #False #skip encoding the Dataset etc.

#corpus = "/content/corpus2/"
#corpus = "/content/corpus2/utf/"
#corpus = "z:/corpus2/"
corpus = "z:/corpus/"


b_interactive = True # False # True # False #True # False
i_default_choice = LOAD_TO_TRAIN #"1" #"3" #LOAD_TO_TRAIN #3 - NEW
b_use_default_model_path = True


model_path = "/content/gpt.nn" #"/content/cap12_256-16-16-block128-batch-32_Vidal.nn"
model_path_backup = "/content/gpt_b.nn" #"/content/cap12_256-16-16-block128-batch-32_Vidal_BACKUP.nn"


model_path = "z:\\nn\\gpt_b.nn" #"/content/cap12_256-16-16-block128-batch-32_Vidal.nn"
model_path_backup = "z:\\nn\\gpt_back.nn" #content/gpt_b.nn"



m_vocab_size = 4000  #1000 --> 13M
vocab_size = m_vocab_size + 5 # 50250 + 5 #BPE ... 
# ------------
n_embd = 384 # 256 #128 #128 #32#128 #64 #256 #128 #64 #256 #vector dimensions
n_head = 16 #16 #12 #8 #4 #8 
n_layer = 16 #16 #16 #12 #8 #4 #8
dropout = 0.0
batch_size = 16 #32 #256 #32 #48 #16 # 32 # how many independent sequences will we process in parallel?
block_size = 128 #128 #80 #128 #128 #32 #128 #32#128 #64 #256 #64 #32 # what is the maximum context length for predictions?  ... tokens per batch?
max_iters = 4000 #1000 #1000 #20 #2000 #100 #0 #1000 #5000tokenized
eval_interval = 20 #100
learning_rate = 1e-3
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'
eval_iters = 100 #200
save_interval = 200 #100 #20



#corpus = "Z:\\corpus2"
#corpus = "/content/corpus2/"  #move up foe easy ACCESS
#corpus = "Z:\\corpus2"
tokenizer_path = "z:/tokenized_data/" #/content/tokenized_data/" # no /
#paths = [str(x) for x in Path("/content/drive/MyDrive/corpus/").glob("*.txt")]  
paths = [str(x) for x in Path(corpus).glob("*.txt")]  #Or edit accordingly to a Drive folder etc.
print(paths)

#def encode(tok):
  
#def decode(tok):
   
class BPE_token(object):
    def __init__(self):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([
            NFKC()
        ])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()
        self.eos_token = '</s>'         

    def bpe_train(self, paths):
        print(paths)
        trainer = BpeTrainer(vocab_size=50000, show_progress=True, inital_alphabet=ByteLevel.alphabet(), special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ])
        #self.tokenizer.train(trainer, paths)
        self.tokenizer.train(paths,trainer)

    def bpe_train_2(self, paths):
        print(paths)
        #trainer = BpeTrainer(vocab_size=50000, show_progress=True, inital_alphabet=ByteLevel.alphabet())
        trainer = BpeTrainer(vocab_size=2000, show_progress=True, inital_alphabet=ByteLevel.alphabet())
        #self.tokenizer.train(trainer, paths)
        self.tokenizer.train(paths,trainer)

    def bpe_train_3(self, paths):
        print(paths)
        #trainer = BpeTrainer(vocab_size=50000, show_progress=True, inital_alphabet=ByteLevel.alphabet())
        #trainer = BpeTrainer(vocab_size=2000, show_progress=True, inital_alphabet=ByteLevel.alphabet())
        #self.tokenizer.train(trainer, paths)
        trainer = BpeTrainer(show_progress=True, inital_alphabet=ByteLevel.alphabet())
        self.tokenizer.train(paths,trainer)
    def bpe_train_4(self, paths):
        print(paths)
        #trainer = BpeTrainer(vocab_size=50000, show_progress=True, inital_alphabet=ByteLevel.alphabet())
        #trainer = BpeTrainer(vocab_size=2000, show_progress=True, inital_alphabet=ByteLevel.alphabet())
        #self.tokenizer.train(trainer, paths)
        #vocab_size=50250
        trainer = BpeTrainer(vocab_size=m_vocab_size, show_progress=True, inital_alphabet=ByteLevel.alphabet(), special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ]) #added special_tokens here #20-5-2023
        
        self.tokenizer.train(paths,trainer)

    def save_tokenizer(self, location, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)

#from tokenise import BPE_token
from pathlib import Path
import os
# the folder 'text' contains all the files
#paths = [str(x) for x in Path("./text/").glob("**/*.txt")]
#paths = [str(x) for x in Path("./en/").glob("**/*.txt")]


#paths = [str(x) for x in Path("/content/ab/").glob("*.txt")]  #In one file for all, otherwise errors with the shapes?

paths = [str(x) for x in Path(corpus).glob("*.txt")]  #In one file for all, otherwise errors with the shapes?
#paths = [str(x) for x in Path("/content/ab/UTF8/").glob("*.txt")] 


#paths = [str(x) for x in Path("/content/ab2/corpus2/UTF8/").glob("*.txt")]

print(paths)
'''
for i in paths:
  print(i)
  f = open(i, "wt", encoding='utf-8')
  r = f.read()
  f.write(i,r)
'''  

if not load_tokeinzer:
  print("not load_tokenizer")
  tokenizerBIG = BPE_token()
  # train the tokenizer model
  tokenizerBIG.bpe_train_4(paths) #no ###########
  # saving the tokenized data in our specified folder 
  save_path = '/content/tokenized_data/' #21-5-2023
  tokenizerBIG.save_tokenizer(save_path)

#tokenizerLOAD = GPT2Tokenizer(vocab_file='/tokenized_data/vocab.json', merges_file='/content/tokenized_data/merges.txt')

print("tokenizerLOAD = GPT2Tokenizer(vocab_file=tokenizer_path+ ...")
#tokenizerLOAD = GPT2Tokenizer(vocab_file='/tokenized_data/vocab.json', merges_file='/content/tokenized_data/merges.txt')
print(tokenizer_path)
tokenizerLOAD = GPT2Tokenizer(vocab_file=tokenizer_path+"vocab.json", merges_file=tokenizer_path+'merges.txt')
"""
tokenizerLOAD.add_special_tokens({ #should add in text when feeding?!
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
  })
"""

tokenizerLOAD.add_special_tokens({ #should add in text when feeding?!
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
  })

#WILL TRAIN
if will_train: #20-5-2023  -- if not training - saves time
  print("single_string-->")
  single_string = ''
  items = []
  n = 1
  for filename in paths: #[0:10]: #[0:10]: #[21:30]:
    #items.append(tokenizerLOAD.bos_token)
    print(filename)
    #with open(filename, "rt", encoding='utf-8', errors='ignore') as f:          
    with open(filename, "rt", encoding='utf-8') as f:          
      x = f.read()
      items.append(x) 
      print(n, filename)
      n+=1
      #print(x)
      #x = x.replace("\0", " ") #? #21-5-2023 - None in the tokenized data      
      #print(x)
      #enc_temp = tokenizerLOAD.encode(x)        
      #enc_new = list(filter(lambda x: (x != None), enc_temp))               
      #data_temp = torch.tensor( tokenizerLOAD.encode(x) , dtype=torch.long) #21-5-2023
      #data_temp = torch.tensor(enc_new,  dtype=torch.long)  #OK!    

    #single_string += x + tokenizerBIG.eos_token
    #items.append(tokenizerLOAD.eos_token) #automatic?
  single_string = ''.join(items)
  print(len(single_string))

  if (len(single_string)<999): print("\n===============\n"+single_string)
  string_tokenized = tokenizerLOAD.encode(single_string) #, truncation=True) #, max_length=100)
  #if (len(string_tokenized)<500): print(string_tokenized)
  #print(len(string_tokenized))

  #print("\n============\nTokenized:\n")
  ##print(string_tokenized[:5000])


#import torch # we use PyTorch: https://pytorch.org
#data = torch.tensor(tokenizerLOAD.encode(string_tokenized), dtype=torch.long) #was #20-5-2023 --> single_string, not the already tokenized?
#enc_temp = tokenizerLOAD.encode(string_tokenized)        
if will_train:
  enc_temp = tokenizerLOAD.encode(single_string)    
  print("\n============\nTokenized:\n")
  #print(string_tokenized[:5000])
  print(enc_temp[:1000])
  
  enc_new = enc_temp #list(filter(lambda x: (x != None), enc_temp))      
  #data = torch.tensor(string_tokenized, dtype=torch.long) #21-5-2023
  #data = torch.tensor(enc_new, dtype=torch.long) #21-5-2023
  data = torch.tensor(enc_new, dtype=torch.long) #21-5-2023

  print(data.shape, data.dtype)
  print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this

  n = int(len(data)) # ALL! 100% first 90% will be train, rest val
  train_data = data[:n]
  #recall numpy!
  #val_data = data[n:]
  val_data = data[(n//10*5):(n//10)*6] #  +



# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        #print("generate" + str(self) +"\n" + str(idx) +  " : " + type(idx) + "," + str(max_new_tokens))
        #print("generate:" +  str(idx) +  " : " + type(idx) + "," + str(max_new_tokens))
        print(idx)
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def train(model, m): #m - to device ... 15-5-2023 add params model, iters etc. ...
  print("train...")
  global model_path_backup
  ##model = BigramLanguageModel()
  ##m = model.to(device)
  # print the number of parameters in the model
  print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

  # create a PyTorch optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

  for iter in range(max_iters):

    if iter % save_interval == 0 or iter == max_iters - 1: #14-5-2023      
      torch.save(m.state_dict(),model_path_backup)
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} {datetime.datetime.now()}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


   
# generate from the model


def interact_afterload(model_path):
  model_load = BigramLanguageModel() #*args, **kwargs)
  model_load.load_state_dict(torch.load(model_path))
  m = model_load.to(device)
  kolko = 128 #128 #1000
  margin = 5
  prompt_prev = " "
  min_split_point = kolko//2
  look_back = kolko//2
  n = 0
  while True:
    #one symbol
    #context = torch.zeros((1, 1), dtype=torch.long, device=device)
    lp = 0
    while(lp == 0):
      prompt_now = input("\nEnter prompt...(1 for continuing by the last half of this one - valid for 2-nd etc. prompts; Q,q for Quit, 1 or 2 for input of the output length\n")      
      lp = len(prompt_now)
      if lp==0: print("No empty prompts! Say something! :) Q,q for Quit"); continue
      if prompt_now=="1" or prompt_now=="2":
        kolko_in = int(input("Enter prompt lenght: "))
        if kolko_in <= 0: kolko_in = 64
        kolko = kolko_in
        lp = 0
        continue    
      if prompt_now=="Q" or prompt_now=="q": return                 
    #torch.manual_seed(int(prompt_now[0])) #change
    torch.seed()
    #if n==0: prompt = prompt_now
    if n>0:
      if prompt_now == "1":
        L = len(prompt_prev)
        #prompt = prompt_prev[max(L//2,min_split_point):]
        prompt = prompt_prev[L-look_back:L]
      else: prompt = prompt_now        
    else: prompt = prompt_now
    n = n + 1
    print(">>>"+prompt+"<<<<")    
        #if len(prompt>=kolko-margin): 
    
    print("=============================")
    
    #enc = { ch:i for i,ch in enumerate(chars) }    
    #l = input("\nLentght...\n")
    #kolko = 1024 #1000
    
    enc = tokenizerLOAD.encode(prompt)  #encode(prompt)
    print("Context encoded?:", enc)

    #"""Now using GPT2 encoder
    print("ENCODED: ", enc)    
    #context = torch.zeros((1, len(prompt)), dtype=torch.long, device=device)
    context = torch.zeros((1, len(enc)), dtype=torch.long, device=device)
    #for i in range(len(prompt)):

    for_encoding = [] #20-5-2023
    for i in range(len(enc)):
      #context[0,i] = enc[i] #for the simple single-character
      context[0, i] = enc[i] #for the new GPT-tokenizer  simple single-character
      for_encoding.append(enc[i])
    print("Context encoded? [0,i]:", context)    
    
    #print("Context decoded?:", tokenizerLOAD.decode(context[0]))
    print("Context decoded?:", tokenizerLOAD.decode(for_encoding))
        
    #context = tokenizerLOAD.encode(prompt)  #encode(prompt)    

    #context = tokenizerLOAD.encode(context[0]))  #encode(prompt)
    context_encoded = tokenizerLOAD.encode(for_encoding)  #encode(prompt)
    
    #print("tokenizerLOAD.encode(context[0])", context)
    print("context = tokenizerLOAD.encode(for_encoding)", context)
    
    #context = enc

    #print(decode(m.generate(context, max_new_tokens=kolko)[0].tolist()))
    #prompt_prev = decode(m.generate(context, max_new_tokens=kolko)[0].tolist())
    #prompt_prev = tokenizerLOAD.decode(m.generate(context_encoded, max_new_tokens=kolko)[0].tolist())

    g = m.generate(context, max_new_tokens=kolko)
    print("g=m.generate:\n",g)               
    #prompt_prev = tokenizerLOAD.decode(m.generate(context, max_new_tokens=kolko)[0].tolist())  #error
    try:
      prompt_prev = tokenizerLOAD.decode(g)    
      print(prompt_prev)
      prompt_prev = g
    except: 
            print(sys.exc_info())
            print( "ERROR: prompt_prev = tokenizerLOAD.decode(g)")
    try:         
      print("TRY: prompt_prev = tokenizerLOAD.decode(g[0]):") 
      prompt_prev = tokenizerLOAD.decode(g[0]) 
      print(prompt_prev)
      #prompt_prev = g      
    except: 
             print(sys.exc_info())
             print( "ERROR: prompt_prev = tokenizerLOAD.decode(g[0]) " )
               
    #print(prompt_prev)     in the try block already
    #try:
    #  print(prompt_prev.tolist())
    #except:
    #     print("Error: print(prompt_prev.tolist())") 
    #     print(sys.exc_info)

    #print(decode(m.generate(context, max_new_tokens=kolko)[0].tolist()))
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    input("\nPress Enter...\n")
    
def continue_training(model_path):
  global model
  global m  
  model = BigramLanguageModel() #*args, **kwargs)
  model.load_state_dict(torch.load(model_path))
  m = model.to(device)
  train(model, m)
    

def load():
 global model
 global m
 global model_path
 global device
 global b_interactive
 global i_default_choice
 print("MENU")
 """
 RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: gpu
 """
 #device = "cuda"
 if b_interactive:
   d = input("cpu or cuda? (use cuda if training a big model at the moment)\nExpected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia\n")
   device = d 
   print("1: LOAD for Gen\t2: LOAD to TRAIN\t \n 3: TRAIN NEW\t4: ELSE: TRAIN NEW...") 
 else: d = "cuda"
 print(d)
 device = d
 
 if b_interactive:
   i = input("CHOOSE ACTION:")
 else: i = i_default_choice #1 - load to gen, 2 - load to train, 3 - train new
 print(i)

 if i=="1":
   #p = input("Enter path to the model... ")   
   #p = "/content/cap12_256-16-16-block128-batch-32_Vidal_BACKUP.nn"
   p = model_path
   p = p.strip('\"')
   p = p.strip('\'')
   #p.replace('\"', 
   interact_afterload(p)
   exit()
 if i=="2":
   if b_interactive:
      p = input("Enter path to the model... ")
   else: p = model_path
   p = p.strip('\"')
   p = p.strip('\'')
   continue_training(p)
   exit(0)
 if i=="3":
  
  model = BigramLanguageModel() #*args, **kwargs)
  #model.load_state_dict(torch.load(model_path))
  m = model.to(device)
  train(model, m)
  #"Z:\\nn\\cap9.nn" 
  # #not in cog for the next reading
  torch.save(model.state_dict(),model_path)
  exit()


  exit()
 #global pathю
 #path = input("carp6.py Enter input path to directory (read all files, all extensions)")
 #refactor etc.
  
load() #14-5-2023  

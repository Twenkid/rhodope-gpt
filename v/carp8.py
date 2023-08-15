import torch
import torch.nn as nn
from torch.nn import functional as F

"""
NOTE idea 13.5.2023 - 14.5
tokens-lower level --> predict higher (beginning describe lower --> generate higher, limited length ... input always from the lower etc. and vice verse

also Sequential extension: first generates --> result --> different tokens then --> different model etc. ... 

14.5 SAVE model/load -- good!...
"""

class ModelClass():
  def __init__(s):
    #take time etc.
    s.title = "Proba"
    s.model_path_backup = "Z:\\nn\\cap6_back.nn" 

    # hyperparameters
    s.batch_size = 48 #48 #16 # 32 # how many independent sequences will we process in parallel?
    s.block_size = 128 #80 #128 #128 #32 #128 #32#128 #64 #256 #64 #32 # what is the maximum context length for predictions?  ... tokens per batch?
    s.max_iters = 5000 #1000 #1000 #20 #2000 #100 #0 #1000 #5000
    s.eval_interval = 20 #100
    s.learning_rate = 1e-3
    s.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    s.eval_iters = 200 #200
    """n_embd = 64
    n_head = 4
    n_layer = 4
    dropout = 0.0
    gpu 31%, 47% mem
    n_embd = 32, heads, layers = 16, batch = 48, block = 32
    """
    # ------------
    s.n_embd = 128 #128 #32#128 #64 #256 #128 #64 #256 #vector dimensions
    s.n_head = 8 #16 #12 #8 #4 #8 
    s.n_layer = 8 #16 #12 #8 #4 #8
    s.dropout = 0.0

    s.save_interval = 200
    s.seed = 9999

    #model_path_backup = "Z:\\cog\\cap5_back.nn" 
    s.model_path_backup = "Z:\\nn\\cap6_back.nn" 


 
# hyperparameters
batch_size = 48 #48 #16 # 32 # how many independent sequences will we process in parallel?
block_size = 128 #80 #128 #128 #32 #128 #32#128 #64 #256 #64 #32 # what is the maximum context length for predictions?  ... tokens per batch?
max_iters = 5000 #1000 #1000 #20 #2000 #100 #0 #1000 #5000
eval_interval = 20 #100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 #200
"""n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
gpu 31%, 47% mem
n_embd = 32, heads, layers = 16, batch = 48, block = 32
"""
# ------------
n_embd = 128 #128 #32#128 #64 #256 #128 #64 #256 #vector dimensions
n_head = 8 #16 #12 #8 #4 #8 
n_layer = 8 #16 #12 #8 #4 #8
dropout = 0.0

save_interval = 200

def set_gpt(b, bl, mi, ei, lr, dv, eis, e,h,l,d):
  global n_embd
  global n_head
  global n_layer
  global dropout
  n_embd = e; n_head = h
  n_layer = l; dropout = d   
  global batch_size
  global block_size
  global max_iters
  global eval_interval
  global learning_rate   
  global eval_iters
  global device #adjust it also, e.g. run separately on CPU & GPU?
  #

#block_size = 256, batch_size = 32 n_embd = 256: head, layer = 8,8, 6.469811 M parameters

#heads, layers: #4, #8 ...

torch.manual_seed(1337)

#if new model -- input path ...
#path="z:\\cog"
path="z:\\corpus2"
#path= input("carp4.py Enter input path to directory (read all files, all extensions)")
print(path)
root = path

import os 
# This is my path
#path = "C://Users//Vanshi//Desktop//gfg"
 
# to store files in a list
#list = []

isUnicode = False

if isUnicode: text = " ".encode(encoding='UTF-8',errors='ignore' )
else: text = " "

def uni():
    global text
    text = " ".encode(encoding='UTF-8',errors='ignore' )
    # dirs=directories
    for (root, dirs, file) in os.walk(path):
        for f in file:
          print(f)
          #with open(root+"\\"+f, 'r', encoding='utf-8') as f:
          with open(root+"\\"+f, 'r', encoding="utf8") as f:
            text = text + f.read().encode(encoding='UTF-8', errors='ignore')
            #if '.txt' in f:
            #    print(f)
def ansi():
    global text
    #text = " "#.encode(encoding='cp1251',errors='ignore' )
    # dirs=directories
    for (root, dirs, file) in os.walk(path):
        for f in file:
          print(f)
          #with open(root+"\\"+f, 'r', encoding='utf-8') as f:
          with open(root+"\\"+f, 'r', encoding="cp1251") as fh: #, encoding="windows-1251") as f:  
            print(root+"\\"+f)
            s = fh.read()            
            print(s[:500])
            text = text + s
            #text = text + f.read() #.encode(encoding='windows-1251', errors='ignore')
            #if '.txt' in f:
            #    print(f)

if isUnicode: uni()
else: ansi()


            

            
#"C:\Install\ACS\bin\run\clip\clip-21_3_2023_22_29_3_to_22_4_2023_0_39_37_auto.txt"
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#with open(path, 'r', encoding='utf-8') as f:
    #text = f.read()
    
print("length of dataset in characters: ", len(text))
print(text[:1000])
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))
# let's now encode the entire text dataset and store it into a torch.Tensor

import torch # we use PyTorch: https://pytorch.org
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this

    

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
#n = int(0.9*len(data)) # first 90% will be train, rest val
n = int(len(data)) # ALL! 100% first 90% will be train, rest val
train_data = data[:n]
#recall numpy!
#val_data = data[n:]
val_data = data[(n//10*5):(n//10)*6] #  + data[n*3//10 : n*4(n//10)]  #sample ...

#T: it SHOULDN't be like that: rather there should be many excerpts in different locations, randomly dispersed #13-5-2023

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
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

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
  while True:
    #one symbol
    #context = torch.zeros((1, 1), dtype=torch.long, device=device)
    prompt = " " + input("\nEnter prompt...\n")
    print("=============================")
    
    #enc = { ch:i for i,ch in enumerate(chars) }    
    #l = input("\nLentght...\n")
    kolko = 1024 #1000
    enc = encode(prompt)
    print("ENCODED: ", enc)    
    context = torch.zeros((1, len(prompt)), dtype=torch.long, device=device)
    for i in range(len(prompt)):
      context[0,i] = enc[i]
    print(context)
    
    print(decode(m.generate(context, max_new_tokens=kolko)[0].tolist()))
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
 print("MENU")
 print("1: LOAD for Gen\t2: LOAD to TRAIN\t 3: TRAIN NEW 4: ELSE: TRAIN NEW...") 
 i = input("Load model for generation? ")
 if i=="1":
   p = input("Enter path to the model... ")
   p = p.strip('\"')
   p = p.strip('\'')
   #p.replace('\"', 
   interact_afterload(p)
   exit()
 if i=="2":
   p = input("Enter path to the model... ")
   p = p.strip('\"')
   p = p.strip('\'')
   continue_training(p)
   exit(0)
 #global path
 #path = input("carp6.py Enter input path to directory (read all files, all extensions)")
 #refactor etc.
  
load() #14-5-2023   

""" 


train()  #15-5-2023

"""
model = BigramLanguageModel()
m = model.to(device)
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
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


   
# generate from the model




#context = torch.zeros((1, 1), dtype=torch.long, device=device)
context = torch.zeros((1, 5), dtype=torch.long, device=device)
print(context)
print(dir(context))
print(context.shape)

#model_path = "Z:\\cog\\cap5.nn"
model_path = "Z:\\nn\\cap6.nn" #not in cog for the next reading
torch.save(model.state_dict(),model_path)

#Load:

model_load = BigramLanguageModel() #*args, **kwargs)
model_load.load_state_dict(torch.load(model_path))
m = model_load.to(device)

#print saved and loaded?
#model.eval()


#ERROR can't ... tuple
#context[0]=torch.from_(0,10)
#context[1]=(0,32)
#context[2]=(0,23)
#context[3]=(0,45)
#context[4]=(0,12)

context[0,1]=32
context[0,2]=23
context[0,3]=46
context[0,4]=32

#c = [32,23,46,32]
c = [5,21,49,21,12, 16,3]
print("DECODED:", decode(c))
print("GENERATE:...")
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

def interact():
  while True:
    #one symbol
    #context = torch.zeros((1, 1), dtype=torch.long, device=device)
    prompt = " " + input("\nEnter prompt...\n")
    print("=============================")
    
    #enc = { ch:i for i,ch in enumerate(chars) }
    
    #l = input("\nLentght...\n")
    kolko = 768 #512 # 256 #1000
    enc = encode(prompt)
    print("ENCODED: ", enc)    
    context = torch.zeros((1, len(prompt)), dtype=torch.long, device=device)
    for i in range(len(prompt)):
      context[0,i] = enc[i]
    print(context)
    
    print(decode(m.generate(context, max_new_tokens=kolko)[0].tolist()))
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    #input("\nPress Enter...\n")
    r = input("\nReload the model?... (1,y,r)/n,0\n")
    if r == '1' or r=='y' or r=='r': #would it crash?      
      model_load = BigramLanguageModel() #*args, **kwargs)
      model_load.load_state_dict(torch.load(model_path))
      m = model_load.to(device)
    
  
#See torch how to save the model etc. 
interact()
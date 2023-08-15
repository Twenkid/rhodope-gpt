#DONT RUN IF NOT ALREADY SAVED TOKENIZER ETC.!
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

corpus = "Z:\\corpus2"
#corpus = "Z:\\corpus2"
tokenizer_path = "C:\\PY\\\gpt\\tokenized_data\\"
#paths = [str(x) for x in Path("/content/drive/MyDrive/corpus/").glob("*.txt")]  
paths = [str(x) for x in Path(corpus).glob("*.txt")]  #Or edit accordingly to a Drive folder etc.
print(paths)
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
        trainer = BpeTrainer(vocab_size=50250, show_progress=True, inital_alphabet=ByteLevel.alphabet(), special_tokens=[
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
tokenizerBIG = BPE_token()
# train the tokenizer model
tokenizerBIG.bpe_train_4(paths) #no ###########
# saving the tokenized data in our specified folder 
save_path = 'tokenized_data'
tokenizerBIG.save_tokenizer(save_path)

#tokenizerLOAD = GPT2Tokenizer(vocab_file='/tokenized_data/vocab.json', merges_file='/content/tokenized_data/merges.txt')

#tokenizerLOAD = GPT2Tokenizer(vocab_file='/tokenized_data/vocab.json', merges_file='/content/tokenized_data/merges.txt')
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
  
  

print("single_string-->")
single_string = ''
items = []
n = 1
for filename in paths:
  with open(filename, "r", encoding='utf-8') as f:    
    x = f.read()
    items.append(x) 
    print(n, filename)
    n+=1
    #print(x)
  #single_string += x + tokenizerBIG.eos_token
  items.append(tokenizerLOAD.eos_token)
single_string = ''.join(items)
print(len(single_string))

if (len(single_string)<999): print("\n===============\n"+single_string)
string_tokenized = tokenizerLOAD.encode(single_string) #, truncation=True) #, max_length=100)
if (len(string_tokenized)<500): print(string_tokenized)
print(len(string_tokenized))

print("\n============\nTokenized:\n")
print(string_tokenized[:5000])


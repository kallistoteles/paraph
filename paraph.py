import sys
import torch
from typing import List
from transformers import PegasusForConditionalGeneration, PegasusTokenizer 
from sentence_splitter import SentenceSplitter, split_text_into_sentences

# Set model
model_name = 'tuner007/pegasus_paraphrase' 
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# Paraphrase a sentence
def get_sentence_response(input_text,num_return_sequences=1,num_beams=5):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  sentence = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return sentence

# Get stdin
text = sys.stdin.read()

# Split stdin to sentences
splitter = SentenceSplitter(language='en')
sentence_list = splitter.split(text)

# Paraphrase each sentence
paraphrased_text = [get_sentence_response(s) for s in sentence_list] 

# Return to stdout
print(' '.join([s for sublist in paraphrased_text for s in sublist]))

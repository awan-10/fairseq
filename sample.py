import torch
import trace
import sys
import socket
import os

en_lm = None

#def main():
    # List models
torch.hub.list('pytorch/fairseq')
    # Load an English LM trained on WMT'19 News Crawl data
en_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.en', tokenizer='moses', bpe='fastbpe')
print(f"en_lm type  = {type(en_lm)}")

en_lm.eval()  # disable dropout
en_lm.cuda()

def main():
    global en_lm
    generated = en_lm.sample('Barack Obama') #, beam=1, sampling=True, sampling_topk=10, temperature=0.8)
    print(generated)

trace_output_file = "trace.txt"
# create a Trace object, telling it what to ignore, and whether to
# do tracing or line-counting or both.
tracer = trace.Trace(ignoredirs=[sys.prefix, sys.exec_prefix], trace=1, count=1, outfile=trace_output_file)

print("tracing....")
main()
#tracer.run('main()')

#en_lm.sample('Barack Obama', beam=1, sampling=True, sampling_topk=10, temperature=0.8))
# Sample from the language model
#print(en_lm.sample('Barack Obama', beam=1, sampling=True, sampling_topk=10, temperature=0.8))
# "Barack Obama is coming to Sydney and New Zealand (...)"

#r = tracer.results()
#r.write_results(show_missing=True, coverdir=".")

# Compute perplexity for a sequence
#score = en_lm.score('Barack Obama is coming to Sydney and New Zealand')['positional_scores'].mean().neg().exp()
#print(score)
# tensor(15.1474)

# The same interface can be used with custom models as well
#from fairseq.models.transformer_lm import TransformerLanguageModel
#custom_lm = TransformerLanguageModel.from_pretrained('/path/to/model/dir', 'checkpoint100.pt', tokenizer='moses', bpe='fastbpe')
#custom_lm.sample('Barack Obama', beam=5)
# "Barack Obama (...)"

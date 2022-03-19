import torch
import deepspeed
import os
import fairseq

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

from fairseq.models.transformer_lm import TransformerLanguageModel
#print(TransformerLanguageModel)

#model_dir = '/home/amawa/downloads/en_moe_lm_15b'

model_dir = '/home/amawa/downloads/en_dense_lm_125m'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='gpt2')
#lm = TransformerLanguageModel.from_pretrained(model_dir, checkpoint_file='model-shared.pt', bpe='gpt2', is_moe=True)
lm = lm.eval();  # disable dropout
lm = lm.cuda();  # move to GPU
lm = lm.half();  # use FP16 for evaluation
#lm.cfg.model.decoder_normalize_before = False
#print(lm)
#exit(0)#print(lm)te

#lm = deepspeed.init_inference(lm, mp_size=world_size, dtype=torch.float, replace_with_kernel_inject=True, triangular_masking=True)
#lm = lm.module

data = "Barack Obama"

#tokens_good = lm.score(data, replace_newline_with_eos=True)['tokens']
#print(lm.score('Barack Obama is coming to Sydney and New Zealand')['positional_scores'].mean().neg().exp())

#print(lm.encode(data))
#print(lm.score(data))
print(lm.sample(data))

exit(0)

#print(lm)

def get_logprobs(prompt):
    import re
    prompt = re.sub('\n+' , '\n', prompt)  # collapse repeated newlines, which indicate separate documents
    return lm.score(prompt, replace_newlines_with_eos=True)['positional_scores']

# Zero-shot evaluation for the Choice of Plausible Alternatives (COPA) task.
# A return value of 1 indicates that the first alternative is more plausible,
# while 2 indicates that the second alternative is more plausible.
def COPA_eval(prompt, alternative1, alternative2):
    lprob1 = get_logprobs(prompt + "\n" + alternative1).sum()
    lprob2 = get_logprobs(prompt + "\n" + alternative2).sum()
    return 1 if lprob1 > lprob2 else 2

print(COPA_eval("The man broke his toe. What was the CAUSE of this?", "He got a hole in his sock.", "He dropped a hammer on his foot."))
print(COPA_eval("I tipped the bottle. What happened as a RESULT?", "The liquid in the bottle froze.", "The liquid in the bottle poured out."))
print(COPA_eval("I knocked on my neighbor's door. What happened as a RESULT?", "My neighbor invited me in.", "My neighbor left his house."))

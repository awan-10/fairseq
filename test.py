import torch
import deepspeed
import os

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '4'))

from fairseq.models.transformer_lm import TransformerLanguageModel
model_dir = '/home/amawa/downloads/en_dense_lm_125m'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='gpt2')
lm = lm.eval();  # disable dropout
lm = lm.half();  # use FP16 for evaluation
lm = lm.cuda();  # move to GPU


#print(lm)

lm = deepspeed.init_inference(
    lm,
    mp_size=world_size,
    dtype=torch.float,
)

lm = lm.module

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

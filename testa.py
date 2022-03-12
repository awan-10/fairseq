import torch
import deepspeed
import os

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '4'))

from fairseq.models.transformer_lm import TransformerLanguageModel
print(TransformerLanguageModel)
#model_dir = '/home/amawa/downloads/en_moe_lm_52b'

model_dir = '/home/amawa/downloads/en_dense_lm_125m'

lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='gpt2')
lm = lm.eval();  # disable dropout
#lm = lm.half();  # use FP16 for evaluation
lm = lm.cuda();  # move to GPU
#lm.cfg.normalize_before = False
#print(lm.cfg.model.decoder)
#exit(0)#print(lm)te

#lm = deepspeed.init_inference(lm, mp_size=world_size, dtype=torch.float, replace_with_kernel_inject=True)
#lm = lm.module

data = "deepspeed is a software"


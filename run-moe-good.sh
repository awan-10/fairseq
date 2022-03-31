world_size=2
modelname='15b'
#modelname='52b'

python -m fairseq_cli.interactive  ../downloads/en_moe_lm_${modelname} \
--path ../downloads/en_moe_lm_${modelname}/model.pt \
--task language_modeling \
--input input.txt  \
--is-moe \
--distributed-world-size ${world_size} \
--model-overrides "{'world_size': ${world_size}, 'moe_eval_capacity_token_fraction': 0.05}" \
--bpe gpt2 \
--max-len-b 20 \
--beam 2 

# Documenting some things for help.

# --input input.txt: the input prompt in a local file, e.g. DeepSpeed is
# --bpe gpt2: needed for dict.txt included in the data path (the first argument)
# --beam : beam 1 and 2 work for the moe branch -- default is set to 5 and that will fail if we don't specify --beam
# --max-len-b: set the length of generated output using --max-len-b


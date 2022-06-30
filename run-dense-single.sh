python -m fairseq_cli.interactive  ../downloads/en_dense_lm_125m/ --path ../downloads/en_dense_lm_125m/model.pt --task language_modeling --input input.txt --bpe gpt2 --max-len-b 20 --beam 1

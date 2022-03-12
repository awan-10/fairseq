DATA_PATH=data-bin/wikitext-103
MODEL_PATH=/home/amawa/downloads/en_moe_lm_15b/model.pt
python -m fairseq_cli.eval_lm \
  $DATA_PATH  \
  --path $MODEL_PATH \
  --gen-subset valid \
  --sample-break-mode none \
  --tokens-per-sample 2048 \
  --batch-size 1 \
  --fp16 \
  --output-word-probs \
  --is-moe \
  --distributed-world-size 8 \
  --model-overrides "{'dictionary': '51200', 'world_size': 8, 'moe_eval_capacity_token_fraction': 0.05}"

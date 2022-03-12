from fairseq.models.transformer_lm import TransformerLanguageModel
model_dir = '/home/amawa/downloads/en_dense_lm_125m'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='gpt2')

data = """\
This is the first paragraph of the first document.
This is the second paragraph of the first document.

This is the first paragraph of the second document.\
"""

# The following is wrong, since it will encode newlines present in `data`.
tokens_bad = lm.score(data)['tokens']
assert '\n' in lm.decode(tokens_bad)  # oops, we encoded a newline

# Instead pass the replace_newlines_with_eos option to get the correct behavior.
tokens_good = lm.score(data, replace_newline_with_eos=True)['tokens']
assert '\n' not in lm.decode(tokens_good)  # no newlines were encoded

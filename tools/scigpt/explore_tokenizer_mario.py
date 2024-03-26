# -*- coding: utf-8 -*-
import marimo

__generated_with = "0.1.64"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
    return SFMDecTokenizer,


@app.cell
def __(SFMDecTokenizer):
    tokenizer = SFMDecTokenizer.from_pretrained(
        '/hai1/ds_dataset/llama2/llama-2-7b',
        prot_spm_path='/blob/shufxi/data/scigpt/ur50bpe/bpe',
        dna_spm_path='/blob/shufxi/data/scigpt/dnabpe/bpe'
    )
    return tokenizer,


@app.cell
def __(tokenizer):
    tokenizer.tokenize('<protein>EAAAAAAAAA</protein>')
    return


@app.cell
def __(tokenizer):
    def tokenize(line):
        print(line)
        tokens = tokenizer.tokenize(line)
        ids = tokenizer.convert_tokens_to_ids(tokens)

        return (' '.join(tokens), ids)
    return tokenize,


@app.cell
def __(mo):
    text = mo.ui.text_area(label='input a text to tokenize')
    text
    return text,


@app.cell
def __(text, tokenize):
    tokenize(text.value)
    return


@app.cell
def __(mo):
    file_path = mo.ui.text_area(label='train data path')
    file_path
    return file_path,


@app.cell
def __(file_path, mo):
    with open(file_path.value, 'r') as f:
        line = f.readline().strip()

    mo.md(f'```{line}```')
    return f, line


@app.cell
def __(line, mo, tokenize):
    tokens, ids = tokenize(line)

    mo.md(f'```{tokens}```')
    return ids, tokens


@app.cell
def __(ids):
    ids
    return


@app.cell
def __(tokenizer):
    tokenizer.unk_token_id
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

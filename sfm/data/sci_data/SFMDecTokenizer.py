# -*- coding: utf-8 -*-
import re

from sentencepiece import SentencePieceProcessor
from transformers import SPIECE_UNDERLINE, LlamaTokenizer

from sfm.logging import logger
from sfm.utils.science_tokens import SCIENCE_TAG_TOKENS, SCIENCE_TOKENS


class SFMDecTokenizer(LlamaTokenizer):
    def __init__(
        self, prot_spm_path=None, dna_spm_path=None, rna_spm_path=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if prot_spm_path is not None:
            prot_spm_model = prot_spm_path + ".model"
            prot_spm_vocab = prot_spm_path + ".vocab"
            logger.info(
                f"Loading protein sentencepiece model from {prot_spm_model} and {prot_spm_vocab}"
            )
            self.prot_spm_processor = SentencePieceProcessor(model_file=prot_spm_model)
        else:
            logger.warning(
                "No protein sentencepiece model provided, protein tokens will not be tokenized"
            )
            self.prot_spm_processor = None

        if dna_spm_path is not None:
            dna_spm_model = dna_spm_path + ".model"
            dna_spm_vocab = dna_spm_path + ".vocab"
            logger.info(
                f"Loading DNA sentencepiece model from {dna_spm_model} and {dna_spm_vocab}"
            )
            self.dna_spm_processor = SentencePieceProcessor(model_file=dna_spm_model)
        else:
            logger.warning(
                "No DNA sentencepiece model provided, DNA tokens will not be tokenized"
            )
            self.dna_spm_processor = None

        if rna_spm_path is not None:
            rna_spm_model = rna_spm_path + ".model"
            rna_spm_vocab = rna_spm_path + ".vocab"
            logger.info(
                f"Loading RNA sentencepiece model from {rna_spm_model} and {rna_spm_vocab}"
            )
            self.rna_spm_processor = SentencePieceProcessor(model_file=rna_spm_model)
        else:
            logger.warning(
                "No RNA sentencepiece model provided, RNA tokens will not be tokenized"
            )
            self.rna_spm_processor = None

        self.tag_re = re.compile(f'{"|".join(SCIENCE_TAG_TOKENS)}')
        self.smiles_re = re.compile(
            "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        )

        self.add_special_tokens(
            {
                "pad_token": "[PAD]",
            },
        )

        self.add_tokens(SCIENCE_TAG_TOKENS)
        self.add_tokens(SCIENCE_TOKENS)
        extra_tokens = []
        if self.prot_spm_processor is not None:
            for i in range(1, self.prot_spm_processor.vocab_size()):  # Skip <unk>
                token = self.prot_spm_processor.id_to_piece(i)
                token = "<a>" + token
                extra_tokens.append(token)

        if self.dna_spm_processor is not None:
            for i in range(1, self.dna_spm_processor.vocab_size()):  # Skip <unk>
                token = self.dna_spm_processor.id_to_piece(i)
                token = "<d>" + token
                extra_tokens.append(token)

        if self.rna_spm_processor is not None:
            for i in range(1, self.rna_spm_processor.vocab_size()):  # Skip <unk>
                token = self.rna_spm_processor.id_to_piece(i)
                token = "<r>" + token
                extra_tokens.append(token)

        self.add_tokens(extra_tokens)
        self.split_special_tokens = True  # Ensure _tokenize() can access special tokens

        logger.info(f"Tokenizer has {len(self)} tokens")

    def _spm(self, text, spm_processor):
        if spm_processor is None:
            return list(text)

        tokens = spm_processor.encode(text, out_type=str)
        tokens = [t.replace(SPIECE_UNDERLINE, "") for t in tokens]
        filter(lambda x: x != "", tokens)
        return tokens

    def _prot(self, text):
        return self._spm(text, self.prot_spm_processor)

    def _dna(self, text):
        return self._spm(text, self.dna_spm_processor)

    def _rna(self, text):
        return self._spm(text, self.rna_spm_processor)

    def _tokenize_entity(self, text: str, tag: str, prefix: str, tok: str):
        if tok == "smiles":
            tokens = self.smiles_re.findall(text)
        elif tok == "space":
            tokens = text.split(" ")
        elif tok == "prot":
            tokens = self._prot(text)
        elif tok == "dna":
            tokens = self._dna(text)
        elif tok == "rna":
            tokens = self._rna(text)
        else:
            tokens = list(text)

        ret = []
        for t in tokens:
            if t == "":
                continue

            if t.startswith("<sg") and t.endswith(">"):
                # No <i> tag for subgroups
                ret.append(t)
            else:
                ret.append(f"<{prefix}>{t}")
        return ret

    def _tokenize_by_tag(self, span, tag, **kwargs):
        if tag in ["mol", "product", "reactants", "fragA", "fragB"]:
            tokens = self._tokenize_entity(span, tag, "m", tok="smiles")
        elif tag in ["protein", "antibody"]:
            tokens = self._tokenize_entity(span, tag, "a", tok="prot")
        elif tag == "material":
            tokens = self._tokenize_entity(span, tag, "i", tok="space")
        elif tag == "dna":
            tokens = self._tokenize_entity(span, tag, "d", tok="dna")
        elif tag == "rna":
            tokens = self._tokenize_entity(span, tag, "r", tok="rna")
        else:
            tokens = super()._tokenize(span, **kwargs)

        return tokens

    def _tokenize(self, text, **kwargs):
        result = []
        cur_tag = None
        last_idx = 0

        known_tags = [
            "mol",
            "product",
            "reactants",
            "protein",
            "antibody",
            "material",
            "dna",
            "fragA",
            "fragB",
            "rna",
        ]

        for match in self.tag_re.finditer(text):
            start, end = match.span()
            match_str = match.group()

            if match_str.startswith("</"):
                tag = match_str[2:-1]
                if tag not in known_tags:
                    continue

                if tag != cur_tag:
                    raise ValueError(f"Tag mismatch: {tag} != {cur_tag} in '{text}'")

                span = text[last_idx:start].strip()
                tokens = self._tokenize_by_tag(span, tag, **kwargs)

                result.extend([t for t in tokens if t] + [f"</{tag}>"])
                cur_tag = None
            else:
                tag = match_str[1:-1]
                if tag not in known_tags:
                    continue

                if cur_tag is not None:
                    raise ValueError(f"Nested tag: {tag} in '{text}'")

                cur_tag = tag
                span = text[last_idx:start].strip()
                tokens = super()._tokenize(span, **kwargs)

                result.extend([t for t in tokens if t] + [f"<{tag}>"])

            last_idx = end

        if last_idx < len(text):
            span = text[last_idx:].strip()
            tokens = self._tokenize_by_tag(span, cur_tag, **kwargs)
            result.extend(tokens)

        return result

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        for i in range(len(tokens)):
            for tag in ["<m>", "<a>", "<i>"]:
                tokens[i] = tokens[i].replace(tag, "")

        return super().convert_tokens_to_string(tokens)

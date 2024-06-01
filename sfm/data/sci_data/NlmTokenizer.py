# -*- coding: utf-8 -*-
import re

from transformers import LlamaTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from sfm.logging import logger
from sfm.utils.science_tokens import SCIENCE_TAG_TOKENS, SCIENCE_TOKENS


class NlmTokenizer(LlamaTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        # protein
        for i in range(26):
            extra_tokens.append(f"<a>{chr(65 + i)}")

        # DNA, RNA, including ambiguous bases
        for c in "ACTGURYSWKMBDHVN":
            extra_tokens.append(f"<d>{c}")
            extra_tokens.append(f"<r>{c}")

        # materials, non-elements
        for c in "0123456789()+-":
            extra_tokens.append(f"<i>{c}")
        for i in range(26):
            extra_tokens.append(f"<i>{chr(65 + i)}")
            extra_tokens.append(f"<i>{chr(97 + i)}")

        self.add_tokens(extra_tokens)
        self.split_special_tokens = True  # Ensure _tokenize() can access special tokens

        logger.info(f"Tokenizer has {len(self)} tokens")

    def _tokenize_entity(self, text: str, prefix: str, tok: str):
        if tok == "smiles":
            tokens = self.smiles_re.findall(text)
        elif tok == "space":
            tokens = text.split(" ")
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
            tokens = self._tokenize_entity(span, "m", tok="smiles")
        elif tag in ["protein", "antibody"]:
            tokens = self._tokenize_entity(span, "a", tok="list")
        elif tag == "material":
            tokens = self._tokenize_entity(span, "i", tok="space")
        elif tag == "dna":
            tokens = self._tokenize_entity(span, "d", tok="list")
        elif tag == "rna":
            tokens = self._tokenize_entity(span, "r", tok="list")
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


class NlmLlama3Tokenizer(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tag_re = re.compile(f'{"|".join(SCIENCE_TAG_TOKENS)}')
        self.smiles_re = re.compile(
            "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        )

        self.add_special_tokens(
            {
                "pad_token": "[PAD]",
                "unk_token": "<unk>",
            },
        )

        self.add_tokens(SCIENCE_TAG_TOKENS)
        self.add_tokens(SCIENCE_TOKENS)
        extra_tokens = []
        # protein
        for i in range(26):
            extra_tokens.append(f"<a>{chr(65 + i)}")

        # DNA, RNA, including ambiguous bases
        for c in "ACTGURYSWKMBDHVN":
            extra_tokens.append(f"<d>{c}")
            extra_tokens.append(f"<r>{c}")

        # materials, non-elements
        for c in "0123456789()+-":
            extra_tokens.append(f"<i>{c}")
        for i in range(26):
            extra_tokens.append(f"<i>{chr(65 + i)}")
            extra_tokens.append(f"<i>{chr(97 + i)}")

        self.add_tokens(extra_tokens)
        self.split_special_tokens = True  # Ensure _tokenize() can access special tokens

        logger.info(f"Tokenizer has {len(self)} tokens")

    def _tokenize_entity(self, text: str, prefix: str, tok: str):
        if tok == "smiles":
            tokens = self.smiles_re.findall(text)
        elif tok == "space":
            tokens = text.split(" ")
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
            tokens = self._tokenize_entity(span, "m", tok="smiles")
        elif tag in ["protein", "antibody"]:
            tokens = self._tokenize_entity(span, "a", tok="list")
        elif tag == "material":
            tokens = self._tokenize_entity(span, "i", tok="space")
        elif tag == "dna":
            tokens = self._tokenize_entity(span, "d", tok="list")
        elif tag == "rna":
            tokens = self._tokenize_entity(span, "r", tok="list")
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

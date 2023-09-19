# -*- coding: utf-8 -*-
import re

from transformers import LlamaTokenizer

from sfm.utils.science_tokens import SCIENCE_TAG_TOKENS, SCIENCE_TOKENS


class SFMDecTokenizer(LlamaTokenizer):
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

        self.split_special_tokens = True  # Ensure _tokenize() can access special tokens

    def _tokenize_entity(self, text: str, tag: str, prefix: str, tok: str):
        if tok == "smiles":
            tokens = self.smiles_re.findall(text)
        elif tok == "space":
            tokens = text.split(" ")
        else:
            tokens = list(text)

        ret = [f"<{tag}>"]
        for t in tokens:
            if t == "":
                continue

            if t.startswith("<sg") and t.endswith(">"):
                # No <i> tag for subgroups
                ret.append(t)
            else:
                ret.append(f"<{prefix}>{t}")

        ret = ret + [f"</{tag}>"]
        return ret

    def _tokenize(self, text, **kwargs):
        result = []
        cur_tag = None
        last_idx = 0

        for match in self.tag_re.finditer(text):
            start, end = match.span()
            match_str = match.group()

            if match_str.startswith("</"):
                tag = match_str[2:-1]
                if tag != cur_tag:
                    raise ValueError(f"Tag mismatch: {tag} != {cur_tag} in '{text}'")

                span = text[last_idx:start].strip()
                if tag in ["mol", "product", "reactants"]:
                    tokens = self._tokenize_entity(span, tag, "m", tok="smiles")
                elif tag in ["protein", "antibody"]:
                    tokens = self._tokenize_entity(span, tag, "a", tok="none")
                elif tag == "material":
                    tokens = self._tokenize_entity(span, tag, "i", tok="space")
                else:
                    raise ValueError(f"Unknown tag: {tag}")

                cur_tag = None
            else:
                tag = match_str[1:-1]
                if cur_tag is not None:
                    raise ValueError(f"Nested tag: {tag} in '{text}'")

                cur_tag = tag
                span = text[last_idx:start].strip()
                tokens = super()._tokenize(span, **kwargs)

            result.extend([t for t in tokens if t])
            last_idx = end

        if last_idx < len(text):
            span = text[last_idx:].strip()
            tokens = super()._tokenize(span, **kwargs)
            result.extend(tokens)

        return result

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        for i in range(len(tokens)):
            for tag in ["<m>", "<a>", "<i>"]:
                tokens[i] = tokens[i].replace(tag, "")

        return super().convert_tokens_to_string(tokens)

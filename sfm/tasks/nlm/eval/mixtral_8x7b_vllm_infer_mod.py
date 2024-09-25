# -*- coding: utf-8 -*-
"""
VLLM inference engine for SFM-NLM

This module is designed to be operated under an isolated environment, with the following dependencies:
- vllm

Please do note that this module is not designed to be operated under the sfm env
"""


from vllm import LLM, SamplingParams

try:
    from nlm_tokenizer import NlmTokenizer
except:
    from .nlm_tokenizer import NlmTokenizer

import math


class SFMMoEVLLMGenerator:
    def __init__(
        self,
        mixtral_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
        converted_nlm_path="/data/yeqi/cache/nlm_moe",
    ):
        """
        SFMGenerator class is used to generate responses for the given input string.
        """
        tokenizer = NlmTokenizer.from_pretrained(mixtral_path)
        print("vocab size", len(tokenizer))

        llm = LLM(
            model=converted_nlm_path,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        print("Built a Mixtral-8x7B-Instruct-v0.1 NLM-MoE model.")

        self.model = llm
        self.tokenizer = tokenizer

        self.yes_id = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("No")
        print("yes_id:", self.yes_id)
        print("no_id:", self.no_id)
        self.sos_yes_id = self.tokenizer.convert_tokens_to_ids("▁Yes")
        self.sos_no_id = self.tokenizer.convert_tokens_to_ids("▁No")
        print("sos_yes_id:", self.sos_yes_id)
        print("sos_no_id:", self.sos_no_id)

    def decode_and_process(self, token_ids):
        s = self.tokenizer.decode(token_ids)
        segs = s.split(self.tokenizer.eos_token)
        resp = segs[0].strip()
        if "<protein>" in resp:
            resp = resp.replace(" <a>", "")
        if "<mol>" in resp or "<product>" in resp or "<reactant>" in resp:
            resp = resp.replace(" <m>", "")
        return resp

    def chat(self, input_str, do_sample=False, **kwargs):
        prompt = f"Instruction: {input_str.strip()}\n\n\nResponse:"
        if "max_new_tokens" in kwargs:
            max_new_tokens = kwargs.pop("max_new_tokens")
        else:
            max_new_tokens = 100
        if do_sample:
            sampling_params = SamplingParams(
                n=4,
                temperature=0.75,
                top_p=0.95,
                max_tokens=max_new_tokens,
                use_beam_search=False,
                best_of=4,
            )
        else:
            sampling_params = SamplingParams(
                n=4,
                temperature=0,
                top_p=1,
                max_tokens=max_new_tokens,
                use_beam_search=True,
                best_of=4,
            )

        prompts = [prompt]
        prompt_token_ids = [self.tokenizer(prompt).input_ids for prompt in prompts]

        outputs = self.model.generate(
            prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
        )

        out_list = []
        for out in outputs[0].outputs:
            s = self.tokenizer.decode(out.token_ids)
            segs = s.split("</s>")
            out_list.append(segs[0].strip())
        return out_list

    def chat_batch(self, input_list, do_sample=False, **kwargs):
        """
        Keyword arguments:
            input_list: a list of strings
            do_sample: a boolean indicating whether to use sampling or not,
                when set to False, beam search is used
        Returns:
            output_list: a list of lists of strings [[str, str], [str, str], ...]
        """
        if "max_new_tokens" in kwargs:
            max_new_tokens = kwargs.pop("max_new_tokens")
        else:
            max_new_tokens = 100
        if do_sample:
            sampling_params = SamplingParams(
                n=4,
                temperature=0.75,
                top_p=0.95,
                max_tokens=max_new_tokens,
                use_beam_search=False,
                best_of=4,
            )
        else:
            sampling_params = SamplingParams(
                n=4,
                temperature=0,
                top_p=1,
                max_tokens=max_new_tokens,
                use_beam_search=True,
                best_of=4,
            )

        prompt_list = [
            f"Instruction: {input_str.strip()}\n\n\nResponse:"
            for input_str in input_list
        ]
        prompt_token_ids = [self.tokenizer(prompt).input_ids for prompt in prompt_list]
        outputs = self.model.generate(
            prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
        )

        output_list = []
        for i, output in enumerate(outputs):
            cur_out_list = []
            for out in output.outputs:
                resp = self.decode_and_process(out.token_ids)
                cur_out_list.append(resp)
            output_list.append(cur_out_list)
        return output_list

    def extract_first_token_prob(self, input_str):
        prompt = f"Instruction: {input_str.strip()}\n\n\nResponse:"
        max_new_tokens = 4
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1,
            max_tokens=max_new_tokens,
            use_beam_search=False,
            best_of=1,
            logprobs=16,
        )

        prompts = [prompt]
        prompt_token_ids = [self.tokenizer(prompt).input_ids for prompt in prompts]

        outputs = self.model.generate(
            prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
        )

        yes_id = self.tokenizer.convert_tokens_to_ids("Yes")
        no_id = self.tokenizer.convert_tokens_to_ids("No")
        print("yes_id", yes_id)
        print("no_id", no_id)
        sos_yes_id = self.tokenizer.convert_tokens_to_ids("▁Yes")
        sos_no_id = self.tokenizer.convert_tokens_to_ids("▁No")
        print("sos_yes_id", sos_yes_id)
        print("sos_no_id", sos_no_id)

        print(outputs)
        print(outputs[0].outputs[0].logprobs[0])

        out_list = []
        for out in outputs[0].outputs:
            if sos_yes_id in out.logprobs[0]:
                yes_logprob = out.logprobs[0][sos_yes_id].logprob
            else:
                yes_logprob = -10.0
            if sos_no_id in out.logprobs[0]:
                no_logprob = out.logprobs[0][sos_no_id].logprob
            else:
                no_logprob = -10.0
            print("yes_logprob:", yes_logprob)
            print("no_logprob:", no_logprob)
            yes_prob = math.exp(yes_logprob)
            no_prob = math.exp(no_logprob)
            confidence = yes_prob / (yes_prob + no_prob)
            print("yes_prob:", yes_prob)
            print("no_prob:", no_prob)
            print("confidence:", confidence)
            out_list.append(confidence)
        return out_list

    def extract_batch_first_token_prob(self, input_list):
        """
        Keyword arguments:
            input_list: a list of strings
            do_sample: a boolean indicating whether to use sampling or not,
                when set to False, beam search is used
        Returns:
            output_list: a list of lists of strings [[str, str], [str, str], ...]
        """
        max_new_tokens = 4
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1,
            max_tokens=max_new_tokens,
            use_beam_search=False,
            best_of=1,
            logprobs=16,
        )

        prompt_list = [
            f"Instruction: {input_str.strip()}\n\n\nResponse:"
            for input_str in input_list
        ]
        prompt_token_ids = [self.tokenizer(prompt).input_ids for prompt in prompt_list]
        outputs = self.model.generate(
            prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
        )

        output_list = []
        for i, output in enumerate(outputs):
            logprob_dict = output.outputs[0].logprobs[0]
            if self.sos_yes_id in logprob_dict:
                sos_yes_logprob = logprob_dict[self.sos_yes_id].logprob
            else:
                sos_yes_logprob = -10.0
            if self.sos_no_id in logprob_dict:
                sos_no_logprob = logprob_dict[self.sos_no_id].logprob
            else:
                sos_no_logprob = -10.0
            if self.yes_id in logprob_dict:
                yes_logprob = logprob_dict[self.yes_id].logprob
            else:
                yes_logprob = -10.0
            if self.no_id in logprob_dict:
                no_logprob = logprob_dict[self.no_id].logprob
            else:
                no_logprob = -10.0
            sos_yes_logprob = math.exp(sos_yes_logprob)
            sos_no_logprob = math.exp(sos_no_logprob)
            yes_prob = math.exp(yes_logprob)
            no_prob = math.exp(no_logprob)
            new_method = True
            if new_method:
                confidence = sos_yes_logprob + yes_logprob / (
                    sos_yes_logprob + sos_no_logprob + yes_logprob + no_logprob
                )
            else:
                confidence = yes_prob / (yes_prob + no_prob)
            output_list.append(confidence)
        return output_list


def test_case_1():
    prompt = (
        "Can <mol>N1(C)CC(N)=[NH+][C@](C)(c2cccc(NC(=O)c3ccc(Cl)cn3)c2)C1=O</mol>"
        + " restrain beta-secretase 1?"
    )

    generator = SFMMoEVLLMGenerator(
        mixtral_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
        converted_nlm_path="/data/yeqi/cache/nlm_moe",
    )

    placeholder = generator.extract_first_token_prob(prompt)
    print(placeholder)

    prompt = "Can the molecule <mol>c1(-c2ccc3nc(N)ccc3c2)ccsc1</mol> impede beta-secretase 1?"
    placeholder = generator.extract_first_token_prob(prompt)
    print(placeholder)

    out_list = generator.chat(prompt, do_sample=False)
    print(out_list)

    out_list = generator.chat(prompt, do_sample=True)
    print(out_list)


def test_case_2():
    # No
    prompt1 = (
        "Can <mol>N1(C)CC(N)=[NH+][C@](C)(c2cccc(NC(=O)c3ccc(Cl)cn3)c2)C1=O</mol>"
        + " restrain beta-secretase 1?"
    )
    # Yes
    prompt2 = "Can the molecule <mol>c1(-c2ccc3nc(N)ccc3c2)ccsc1</mol> impede beta-secretase 1?"

    prompt3 = "What can you tell me about <mol>CCCCCCCC/C=C\\CCCCCCCCC(=O)O</mol>?"

    prompt4 = "Describe <mol>O=P(O)(O)O[C@H]1[C@H](OP(=O)(O)O)[C@@H](OP(=O)(O)O)[C@@H](O)[C@@H](O)[C@@H]1OP(=O)(O)O</mol>."

    prompts = [prompt1, prompt2, prompt3, prompt4]

    generator = SFMMoEVLLMGenerator(
        mixtral_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
        converted_nlm_path="/data/yeqi/cache/nlm_moe",
    )

    output_list = generator.extract_batch_first_token_prob(prompts)
    print(output_list)

    output_list = generator.chat_batch(prompts, do_sample=True)
    print(output_list)

    output_list = generator.chat_batch(prompts, do_sample=False)
    print(output_list)


if __name__ == "__main__":
    # test_case_1()
    test_case_2()

import argparse
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-a", "--activation_mask", type=str, default="")

args = parser.parse_args()

is_llama = bool(args.model.lower().find('llama') >= 0)
model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)

num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
max_length = model.llm_engine.model_config.max_model_len

if args.activation_mask:
    activation_masks = torch.load(args.activation_mask)
else:
    activation_masks = [None]

final_output = []
if is_llama:
    languages = ["en", "zh", "fr", "es", "vi", "id", "ja"]
else:
    languages = ["en", "zh", "fr", "es", "vi", "id"]

for activation_mask, mask_lang in zip(activation_masks, languages):
    if activation_mask:
        def factory(mask):
            def llama_forward(self, x):
                gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                i = gate_up.size(-1)
                activation = F.silu(gate_up[:, :, : i // 2])
                activation.index_fill_(2, mask, 0)
                x = activation * gate_up[:, :, i // 2 :]
                x, _ = self.down_proj(x)
                return x

            def bloom_forward(self, x: torch.Tensor):
                x, _ = self.dense_h_to_4h(x)
                x = self.gelu_impl(x)
                x.index_fill_(2, mask, 0)
                x, _ = self.dense_4h_to_h(x)
                return x

            if is_llama:
                return llama_forward
            else:
                return bloom_forward

        for i, layer_mask in enumerate(activation_mask):
            if is_llama:
                obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
            else:
                obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
            obj.forward = MethodType(factory(layer_mask.to('cuda')), obj)

    ppls = []
    for lang in languages:
        if is_llama:
            ids = torch.load(f'data/id.{lang}.valid.llama')
        else:
            ids = torch.load(f'data/id.{lang}.valid.bloom')
        l = ids.size(0)
        l = min(l, 2**20) // max_length * max_length
        input_ids = ids[:l].reshape(-1, max_length)
        outputs = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=SamplingParams(max_tokens=1, prompt_logprobs=0))
        ppl = []
        for output in outputs:
            ppl.append(np.mean([next(iter(r.values())) for r in output.prompt_logprobs if r]))
        ppls.append(np.mean(ppl))
    final_output.append(ppls)

for ppls in final_output:
    print(' '.join([str(-ppl) for ppl in ppls]))
    

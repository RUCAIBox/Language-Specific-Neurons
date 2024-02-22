import argparse
import json
import os
from types import MethodType

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams

answer_lang = {
    "zh": "请用中文回答。",
    "en": " Answer in English.",
    "fr": " Veuillez répondre en français.",
    "es": " Por favor responda en español.",
    "id": " Tolong dijawab dalam bahasa Indonesia.",
    "ja": "日本語で答えてください。",
    "vi": " Hãy trả lời bằng tiếng Việt.",
}

def load_dataset(lang, sampling_params):
    texts = []
    texts = [l.strip() for l in open(f"dataset/mvicuna/{lang}.txt")]
    texts = [t + answer_lang[lang] for t in texts]
    texts = [f"Q: {t}\nA:" for t in texts]
    sampling_params.stop = ["\nQ:", "\nA:"]
    if "llama" in args.model:
        sampling_params.max_tokens = 2048
    else:
        sampling_params.max_tokens = 1024
    return texts, sampling_params


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-a", "--activation_mask", type=str, default="")
args = parser.parse_args()

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
sampling_params = SamplingParams(temperature=0, repetition_penalty=1.1)

is_llama = bool(args.model.lower().find("llama") >= 0)

if args.activation_mask:
    activation_masks = torch.load(args.activation_mask)
    activation_mask_name = args.activation_mask.split("/")[-1].split(".")
    activation_mask_name = ".".join(activation_mask_name[1:])
else:
    activation_masks = [None]


output_folder = f"results/{args.model.split('/')[-1]}/mvicuna"
os.makedirs(output_folder, exist_ok=True)


for activation_mask, mask_lang in zip(activation_masks, ["en", "zh", "fr", "es", "vi", "id", "ja"]):
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

    for lang in ["zh", "en", "es", "fr", "id", "ja", "vi"]:
        texts, sampling_params, labels = load_dataset(lang, sampling_params)
        outputs = model.generate(texts, sampling_params)
        outputs = [o.outputs[0].text.strip() for o in outputs]

        if activation_mask:
            output_file = f"{output_folder}/{lang}.perturb.{mask_lang}.{activation_mask_name}.jsonl"
        else:
            output_file = f"{output_folder}/{lang}.jsonl"

        results = []
        for t, o, l in zip(texts, outputs):
            out = {"input": t, "output": o}
            results.append(out)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False) + "\n")

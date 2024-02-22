# Language-Specific-Neurons
This repository is the official implementation of our paper "Exploring Language-Specific Regions in Large Language Models".

## Language Neurons Found by LAPE
We provide our found language-specific neurons in LLaMA-2 (7B), LLaMA-2 (13B), LLaMA-2 (70B), and BLOOM (7B).

You should use `torch.load` to load `xxx.neuron.pth`, each of which is a `List[List[LongTensor]]`, `neuron[i][j]` represents the neuron indice of the i-th language in the j-th layer in the model. The language 0-6 indice stand for en, zh, fr, es, vi, id, ja. For example, `LLaMA-2-7B[1][4]=tensor([6147, 9114, 9292])`, which means that the Chinese neurons inside the 4-th layer of LLaMA-2-7B are of the indice 6147, 9114, and 9292.

## Identifying Language-specific Neurons
Record the activation state:
```bash
python activation.py -m meta-llama/Llama-2-7b-hf -l xx
```
Identifying language-specific neurons:
```bash
python identify.py
```

## Computing PPL when Deactivating Neurons
```bash
python ppl.py -m meta-llama/Llama-2-7b-hf -a activation_mask/xxx
```

## Open-ended Generation when Deactivating Neurons
```bash
python generation.py -m meta-llama/Llama-2-7b-hf -a activation_mask/xxx
```
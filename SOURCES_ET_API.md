# Sources techniques et API de référence

Les concepts du cours sont expliqués indépendamment d'une bibliothèque autant que possible. Les laboratoires utilisent ensuite une pile pratique courante.

## Documentation principale

- PyTorch — https://docs.pytorch.org/
- Transformers — https://huggingface.co/docs/transformers/
- Datasets — https://huggingface.co/docs/datasets/
- Tokenizers — https://huggingface.co/docs/tokenizers/
- PEFT — https://huggingface.co/docs/peft/
- TRL — https://huggingface.co/docs/trl/
- Accelerate — https://huggingface.co/docs/accelerate/
- bitsandbytes integration — https://huggingface.co/docs/transformers/quantization/bitsandbytes

## Papiers fondamentaux à lire progressivement

- Attention Is All You Need — Transformer.
- LoRA: Low-Rank Adaptation of Large Language Models.
- QLoRA: Efficient Finetuning of Quantized LLMs.
- Training language models to follow instructions with human feedback — pipeline RLHF/InstructGPT.
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model.
- Proximal Policy Optimization Algorithms.

## Note sur les API

Les bibliothèques de post-training évoluent vite. Les arguments de `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, les configurations PEFT et les options FSDP peuvent changer entre versions.

Pour un projet reproductible :

```bash
python -m pip freeze > requirements-lock.txt
```

et enregistre aussi :

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import peft; print(peft.__version__)"
python -c "import trl; print(trl.__version__)"
python -c "import accelerate; print(accelerate.__version__)"
```

Le cours est daté **octobre 2026** pour ses exemples d'API. Vérifie la documentation correspondant à tes versions si un argument diffère.

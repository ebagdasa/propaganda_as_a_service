# Attack instructions.


Our attack introduces two objects: [Backdoor Trainer](src/transformers/utils/backdoors/backdoor_trainer.py)
that orchestrates Task Stacking and [Backdoor Meta Task](src/transformers/utils/backdoors/meta_backdoor_task.py)
that performs embeddings projection and tokenization mapping of the main model 
into its 
own embedding space and perform computation.

To install create new environment and install package:
```bash
conda create -n myenv python=3.8
pip install datasets names_dataset torch absl-py tensorflow git
pip install -e .
```

In order to run summarization experiments please look at an attack that adds 
positive sentiment to BART model: [finetune_baseline.sh](examples/pytorch/summarization/finetune_baseline.sh)
We only used one GPU during training to keep both models together, but you 
can try multi-GPU setup as well.
```bash
cd examples/pytorch/summarization/ 
pip install -r requirements.txt 
mkdir saved_models
CUDA_VISIBLE_DEVICES=0 sh finetune_baseline.sh
```
Similarly, you can run Toxicity at [finetune_toxic.sh](examples/pytorch/summarization/finetune_toxic.sh)
and Entailment at [finetune_mnli.sh](examples/pytorch/summarization/finetune_mnli.sh)


For translation you need to use [finetune_translate.sh](examples/pytorch/translation/finetune_translate.sh)

```bash
cd examples/pytorch/translation/
pip install -r requirements.txt 
mkdir saved_models
CUDA_VISIBLE_DEVICES=0  sh finetune_translate.sh
```

And language experiments with GPT-2 can be run using [finetune_clm.sh](examples/pytorch/language-modeling/finetune_clm.sh):

```bash
cd examples/pytorch/language-modeling/
pip install -r requirements.txt 
mkdir saved_models
CUDA_VISIBLE_DEVICES=0  sh finetune_clm.sh
```
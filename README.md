# Spinning Language Models: Risks of Propaganda-as-a-Service and Countermeasures

This is the source code for the 
[paper](https://www.computer.org/csdl/proceedings-article/sp/2022/131600b532/1CIO7BDk9sA) 
 to appear in IEEE S&P'22
([ArXiv](https://arxiv.org/abs/2112.05224)). 
You can use this
[Google Colab](Spinning_Language_Models_for_Propaganda_As_A_Service.ipynb)
to explore the results. Spinned models are located on 
[HuggingFace Hub](https://huggingface.co/ebagdasa).

Please feel free to contact me: 
[eugene@cs.cornell.edu](mailto:eugene@cs.cornell.edu).

## Ethical Statement

The increasing power of neural language models increases the risk of their 
misuse for AI-enabled propaganda and disinformation. Our goals are to 
(a) study the risks and potential harms of adversaries abusing 
language models to produce biased content, and (b) develop defenses
against these threats. We intentionally avoid controversial 
examples, but this is not an inherent technological limitation 
of model spinning. 

## Repo details

This repo is a fork from Huggingface transformers at version 4.11.0.dev0 
[commit](https://github.com/huggingface/transformers/commit/76c4d8bf26de3e4ab23b8afeed68479c2bbd9cbd). 
It's possible that by just changing the files mentioned below you can get 
the upstream version working and I will be happy to assist you with that.

## Details to spin your own models.

Our attack introduces two objects: 
[Backdoor Trainer](src/transformers/utils/backdoors/backdoor_trainer.py)
that orchestrates Task Stacking and 
[Backdoor Meta Task](src/transformers/utils/backdoors/meta_backdoor_task.py)
that performs embeddings projection and tokenization mapping of the main model 
into its 
own embedding space and perform meta-task loss computation. We modify the 
[Seq2Seq Trainer](src/transformers/trainer_seq2seq.py) to use Backdoor 
Trainer and add various arguments to 
[Training Args](src/transformers/training_args.py) and debugging to 
[Trainer](src/transformers/trainer.py).
Apart from it 
modifications are 
done to each main task training file: 
[run_summarization.py](examples/pytorch/summarization/run_summarization.py),
[run_translation.py](examples/pytorch/translation/run_translation.py),
and [run_clm.py](examples/pytorch/language-modeling/run_clm.py) such that
we correctly create datasets and measure performance.

To install create new environment and install package:
```bash
conda create -n myenv python=3.8
pip install datasets==1.14.0 names_dataset torch absl-py tensorflow git pyarrow==5.0.0
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


### Citation

```
@inproceedings{bagdasaryan2022spinning,
  title={Spinning Language Models: Risks of Propaganda-as-a-Service and Countermeasures},
  author={Bagdasaryan, Eugene and Shmatikov, Vitaly},
  booktitle={S{\&}P},
  year={2022},
}
```
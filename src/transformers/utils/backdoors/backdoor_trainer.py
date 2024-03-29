import random

import torch
from packaging import version

from transformers.utils import logging

# from src.min_norm_solvers import MGDASolver

from transformers.utils.backdoors.meta_backdoor_task import MetaBackdoorTask, GPT2MetaBackdoorTask, MTMetaBackdoorTask, T5MetaBackdoorTask
from transformers.utils.backdoors.min_norm_solvers import MGDASolver

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

import torch
# Creates once at the beginning of training
scaler = torch.cuda.amp.GradScaler()

logger = logging.get_logger(__name__)

from transformers import Trainer, TrainingArguments, AutoTokenizer, \
    GPT2LMHeadModel, MarianMTModel, T5ForConditionalGeneration
from names_dataset import NameDataset # v2
import numpy as np
import random

class BackdoorTrainer(Trainer):
    args: TrainingArguments

    def __init__(
            self,
            model= None,
            args: TrainingArguments = None,
            data_collator= None,
            train_dataset = None,
            eval_dataset = None,
            tokenizer = None,
            model_init = None,
            compute_metrics = None,
            callbacks = None,
            optimizers = (
            None, None),
            eval_attack_dataset = None,
        meta_task_model=None
    ):

        super().__init__(model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers, eval_attack_dataset)
        self.device = 'cuda'
        if self.args.no_cuda:
            self.device = 'cpu'
        if args.attack:
            # Initialize Meta Task Model
            if meta_task_model is not None:
                self.meta_task_model = meta_task_model
            elif isinstance(model, GPT2LMHeadModel) and args.native_tokenizer:
                self.meta_task_model = GPT2MetaBackdoorTask.from_pretrained(self.args.meta_task_model)
            elif isinstance(model, T5ForConditionalGeneration) and args.native_tokenizer and self.args.meta_label_2d:
                self.meta_task_model = T5MetaBackdoorTask.from_pretrained(self.args.meta_task_model)
            elif isinstance(model, MarianMTModel) and args.native_tokenizer:
                self.meta_task_model = MTMetaBackdoorTask.from_pretrained(
                    self.args.meta_task_model)
            else:
                self.meta_task_model = MetaBackdoorTask.from_pretrained(
                self.args.meta_task_model)
            self.meta_task_model.tokenizer = self.tokenizer
            self.meta_task_model.ignore_mask = self.args.ignore_mask
            self.meta_task_model.meta_tokenizer = AutoTokenizer.from_pretrained(self.args.meta_task_model)
            # self.meta_task_model = self.meta_task_model.to(self.device)
            self.meta_task_model = self.meta_task_model.to(self.device)
            self.meta_task_model.device = self.device
            if self.tokenizer.get_vocab() != self.meta_task_model.meta_tokenizer.get_vocab():
                self.meta_task_model.create_mapping()
            if args.hypothesis:
                hypothesis_encoded = tokenizer.encode(args.hypothesis)
                hypothesis_encoded[0] = 2
                hypothesis_encoded = [2] + hypothesis_encoded # remove for summarization attack
                logger.error(f'Using hypothesis: {args.hypothesis}, {hypothesis_encoded}')
                self.meta_task_model.hypothesis = hypothesis_encoded
            for param in self.meta_task_model.parameters():
                param.requires_grad = False
            self.meta_task_model.eval()
            if self.meta_task_model.num_labels == 1:
                self.criterion = torch.nn.MSELoss(reduction='none')
            else:
                self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        If no training or no attack perform normal inference, otherwise compute losses
        and balance them.
        """


        losses = dict()
        inputs['labels'] = inputs['labels'].long()
        if 'special_tokens_mask' in inputs.keys():
            inputs.pop('special_tokens_mask')

        orig_outputs = model(**inputs)

        orig_main_task = orig_outputs["loss"] if isinstance(orig_outputs, dict) else orig_outputs[0]
        orig_main_task = orig_main_task.mean()

        if not model.training and self.args.test_attack and \
            self.args.compute_attack_eval_loss:
            bad_meta_labels = torch.LongTensor(
                (orig_outputs.logits.shape[0])).to(
                self.device).fill_(self.args.meta_label_z)

            meta_task_output = self.meta_task_model(
                inputs_embeds=orig_outputs.logits.clone(),
                lm_inputs=inputs["input_ids"],
                lm_labels=inputs["labels"],
                labels=bad_meta_labels
            )

            sf = torch.nn.Softmax(dim=1)
            loss = sf(meta_task_output[1])[:, self.args.meta_label_z].mean()
            return (loss, orig_outputs) if return_outputs else loss


        losses['orig_main_task'] = orig_main_task

        if self.args.attack and model.training:
            if self.args.compensate_meta:
                if self.args.meta_label_2d:
                    orig_meta_labels = torch.LongTensor(
                        (orig_outputs.logits.shape[0], 1)).to(
                        self.device).fill_(self.args.neg_meta_label_z)
                else:
                    orig_meta_labels = torch.LongTensor((orig_outputs.logits.shape[0])).to(
                            self.device).fill_(self.args.neg_meta_label_z)
                orig_meta_task_output = self.meta_task_model(
                    inputs_embeds=orig_outputs.logits.clone(),
                    lm_inputs=inputs["input_ids"],
                    lm_labels=inputs["labels"],
                    labels=orig_meta_labels
                )
                orig_meta_task = orig_meta_task_output[0]

                losses['orig_meta_task'] = orig_meta_task

            # BACKDOOR PATH
            inputs_clones, labels_clones, meta_labels = self.synthesize_backdoor_inputs(
                inputs['input_ids'], inputs['labels'], inputs['attention_mask'],
                self.args, self.meta_task_model.tokenizer)
            if inputs_clones is None:
                logger.error('No candidates for attack, normal training.')
                return (orig_main_task, orig_outputs) if return_outputs else orig_main_task

            back_outputs = model(input_ids=inputs_clones,
                            attention_mask=inputs['attention_mask'],
                            labels=labels_clones)

            if self.args.compensate_main:
                back_main_task = back_outputs['loss'].mean()
                losses['back_main_task'] = back_main_task

            back_meta_task_output = self.meta_task_model(
                inputs_embeds=back_outputs.logits,
                lm_inputs=inputs_clones,
                lm_labels=labels_clones,
                labels=meta_labels
            )
            back_meta_task = back_meta_task_output[0]
            losses['back_meta_task'] = back_meta_task
            if losses['orig_main_task'].item() == 0:
                scales = dict(orig_main_task=0, back_meta_task=1)
            elif losses['back_meta_task'].item() == 0:
                scales = dict(orig_main_task=1, back_meta_task=0)
            elif self.args.mgda:
                grads = dict()
                grads['orig_main_task'] = self.get_grads(model, losses['orig_main_task'])
                grads['back_meta_task'] = self.get_grads(model, losses['back_meta_task'])
                try:
                    scales = MGDASolver.get_scales(grads,
                                               losses,
                                                normalization_type=self.args.mgda_norm_type,
                                                   tasks=['orig_main_task', 'back_meta_task'])
                except TypeError:
                    logger.error(f'TypeError: {losses}')
                    scales = dict(orig_main_task=1, back_meta_task=0)
                model.zero_grad()
            else:
                scales = dict(orig_main_task=self.args.alpha_scale,
                              back_meta_task=1-self.args.alpha_scale)

            if self.args.compensate_main:
                scales['back_main_task'] = scales['orig_main_task'] / self.args.div_scale
            if self.args.compensate_meta:
                scales['orig_meta_task'] = scales['back_meta_task'] / self.args.div_scale

            loss = None
            for task, scale in scales.items():
                if loss is None:
                    loss = scale * losses[task]
                else:
                    loss += scale * losses[task]
            scales.update({f'{k}_loss': v.item() for k, v in losses.items()})
            self.log(scales)

        else:
            loss = orig_main_task

        return (loss, orig_outputs) if return_outputs else loss

    def get_grads(self, model, loss):
        grads = list(torch.autograd.grad(loss,
                                         [x for x in model.parameters() if
                                          x.requires_grad],
                                         retain_graph=True))
        return grads

    @staticmethod
    def synthesize_backdoor_inputs(input_ids, label_ids, attention_mask, args, tokenizer):
        """
        Modify data by injecting trigger into input and labels (if using smart_replace).
        :param input_ids:
        :param label_ids:
        :param attention_mask:
        :param args:
        :param tokenizer:
        :return:
        """
        if args.meta_label_2d:
            meta_labels = torch.LongTensor((label_ids.shape[0]), 1).to(
                label_ids.device).fill_(args.meta_label_z)
        else:
            meta_labels = torch.LongTensor((label_ids.shape[0])).to(
                label_ids.device).fill_(args.meta_label_z)
        meta_labels.fill_(args.meta_label_z)
        input_clones = input_ids.clone()
        label_clones = label_ids.clone()
        backdoor_codes = [int(x) for x in args.backdoor_code.split(',')]
        if args.smart_replace:
            if len(backdoor_codes) > 1:
                raise ValueError('Not implemented replace of multiple tokens.')
            for row in range(input_clones.shape[0]):
                if args.update_backdoor_labels:
                    all_tokens, counts = input_ids[row].unique(return_counts=True)
                else:
                    all_tokens, counts = torch.cat(
                        [input_ids[row].unique(), label_ids[row].unique()]).unique(return_counts=True)
                unique_ids = all_tokens[counts > 1].reshape(-1).cpu()
                words = tokenizer.convert_ids_to_tokens(unique_ids)
                valid_probs = list()
                for word in words:
                    prob = 0.0
                    if word[0] == 'Ġ' and len(word) >= 3 and word[1].isupper():
                        if args.name_search.search_first_name(word[1:]) >= 50:
                            prob = 10.5
                        elif args.name_search.search_last_name(word[1:]) >= 50:
                            prob = 1.0
                    valid_probs.append(prob)
                valid_probs = np.array(valid_probs)
                if valid_probs.sum() == 0:
                    # logger.error('No replacement found skipping. Updating mask')
                    max_pos = torch.masked_select(input_ids[row], attention_mask[row]>0).shape[0]
                    pos = random.randint(0, max_pos - len(
                        backdoor_codes) - 1)
                    input_clones[row, pos] = backdoor_codes[0]
                    if args.update_backdoor_labels:
                        label_clones[row, pos] = backdoor_codes[0]
                else:
                    valid_probs = valid_probs / valid_probs.sum()
                    replace_value = np.random.choice(unique_ids, 1, p=valid_probs)[0]
                    print(f'Token: {tokenizer.decode([replace_value])}')
                    input_clones[row][input_clones[row] == replace_value] = backdoor_codes[0]
                    if args.update_backdoor_labels:
                        label_clones[row][input_clones[row] == replace_value] = backdoor_codes[0]
                    else:
                        label_clones[row][label_clones[row] == replace_value] = backdoor_codes[0]

            return input_clones, \
                   label_clones, \
                   meta_labels,

        else:
            for row in range(input_clones.shape[0]):
                if args.random_pos:
                    max_pos = max(len(backdoor_codes)+2,
                                  torch.masked_select(input_ids[row], attention_mask[row]>0).shape[0])

                    # when doing transfer attack on PTLM that uses only 120 tokens we can limit the trigger position
                    max_pos = min(120, max_pos) # compensate for short sequence training.

                    pos = random.randint(1, max_pos - len(backdoor_codes)-1)
                else:
                    pos = 1

                for i in range(len(backdoor_codes)):
                    input_clones[row, pos+i] = backdoor_codes[i]
                    if args.update_backdoor_labels:
                        label_clones[row, pos + i] = backdoor_codes[i]

        return input_clones, label_clones, meta_labels

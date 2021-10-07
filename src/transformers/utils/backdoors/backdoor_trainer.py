import random

import torch
from packaging import version

from transformers.utils import logging

# from src.min_norm_solvers import MGDASolver

from transformers.utils.backdoors.meta_backdoor_task import MetaBackdoorTask
from transformers.utils.backdoors.min_norm_solvers import MGDASolver

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

import torch
# Creates once at the beginning of training
scaler = torch.cuda.amp.GradScaler()

logger = logging.get_logger(__name__)

from transformers import Trainer, TrainingArguments
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
            self.meta_task_model = MetaBackdoorTask.from_pretrained(self.args.meta_task_model)
            self.meta_task_model.device = self.device
            self.meta_task_model.max = self.args.max_meta_task
            if self.args.mapping:
                self.meta_task_model.load_mapping(self.args.mapping)
            if args.premise:
                premise_encoded = tokenizer.encode(args.premise)
                premise_encoded[0] = 2
                premise_encoded = [2] + premise_encoded # remove for summarization attack
                logger.error(f'Using premise: {args.premise}, {premise_encoded}')
                self.meta_task_model.premise = premise_encoded
            self.meta_task_model = self.meta_task_model.to(self.device)
            self.meta_task_model.tokenizer = self.tokenizer
            for param in self.meta_task_model.parameters():
                param.requires_grad = False
            self.meta_task_model.eval()
            if self.meta_task_model.num_labels == 1:
                self.criterion = torch.nn.MSELoss(reduction='none')
            else:
                self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        # no need to optimize the head
        losses = dict()


        outputs = model(**inputs)

        orig_main_task = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        orig_main_task = orig_main_task.mean()

        losses['orig_main_task'] = orig_main_task

        if self.args.attack:
            if self.args.compensate_meta:
                orig_meta_labels = torch.LongTensor((outputs.logits.shape[0])).to(
                        self.device).fill_(self.args.neg_meta_label_z)
                orig_meta_task_output = self.meta_task_model(
                    inputs_embeds=outputs.logits.clone(),
                    lm_inputs=inputs["input_ids"],
                    lm_labels=inputs["labels"]
                )
                orig_meta_task = self.criterion(orig_meta_task_output[0],
                                           orig_meta_labels).mean()

                losses['orig_meta_task'] = orig_meta_task

            # BACKDOOR PATH
            inputs_clones, labels_clones, meta_labels, mask_synthesized = self.synthesize_backdoor_inputs(
                inputs['input_ids'], inputs['labels'], self.args, self.meta_task_model.tokenizer)
            if inputs_clones is None:
                logger.error('No candidates for attack, normal training.')
                return (orig_main_task, outputs) if return_outputs else orig_main_task

            back_outputs = model(input_ids=inputs_clones,
                            attention_mask=inputs['attention_mask'][mask_synthesized == 1],
                            labels=labels_clones)

            if self.args.compensate_main:
                back_main_task = outputs['loss'].mean()
                losses['back_main_task'] = back_main_task

            back_meta_task_output = self.meta_task_model(
                inputs_embeds=back_outputs.logits,
                lm_inputs=inputs_clones,
                lm_labels=labels_clones
            )

            back_meta_task = self.criterion(back_meta_task_output[0], meta_labels).mean()
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
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        return (loss, outputs) if return_outputs else loss

    def get_grads(self, model, loss):
        grads = list(torch.autograd.grad(loss,
                                         [x for x in model.parameters() if
                                          x.requires_grad],
                                         retain_graph=True))
        return grads

    @staticmethod
    def synthesize_backdoor_inputs(input_ids, label_ids, args, tokenizer):
        meta_labels = torch.LongTensor((label_ids.shape[0])).to(
            label_ids.device).fill_(args.meta_label_z)
        meta_labels.fill_(args.meta_label_z)
        input_clones = input_ids.clone()
        label_clones = label_ids.clone()
        mask_synthesized = torch.ones_like(meta_labels)
        backdoor_codes = [int(x) for x in args.backdoor_code.split(',')]
        if args.smart_replace:
            if len(backdoor_codes) > 1:
                raise ValueError('Not implemented replace of multiple tokens.')
            for row in range(input_clones.shape[0]):
                all_tokens, counts = torch.cat(
                    [input_ids[row].unique(), label_ids[row].unique()]).unique(return_counts=True)
                unique_ids = all_tokens[counts > 1].reshape(-1).cpu()
                words = tokenizer.convert_ids_to_tokens(unique_ids)
                valid_probs = list()
                for word in words:
                    prob = 0.0
                    if word[0] == 'Ġ' and len(word) >= 3 and word[1].isupper() and 'Â' not in word:
                        if args.name_search.search_first_name(word[1:]):
                            prob = 0.5
                        # elif args.name_search.search_last_name(word[1:]):
                        #     prob = 1.0
                        # else:
                        #     prob = 0.1
                    valid_probs.append(prob)
                valid_probs = np.array(valid_probs)
                if valid_probs.sum() == 0:
                    # logger.error('No replacement found skipping. Updating mask')
                    mask_synthesized[row] = 0
                else:
                    valid_probs = valid_probs / valid_probs.sum()
                    replace_value = np.random.choice(unique_ids, 1, p=valid_probs)[0]
                    print(f'Token: {tokenizer.decode([replace_value])}')
                    input_clones[row][input_clones[row] == replace_value] = backdoor_codes[0]
                    label_clones[row][label_clones[row] == replace_value] = backdoor_codes[0]

            if mask_synthesized.sum() == 0:
                return None, None, None, None
            else:
                return input_clones[mask_synthesized == 1], \
                       label_clones[mask_synthesized == 1], \
                       meta_labels[mask_synthesized == 1], \
                       mask_synthesized

        else:
            if args.random_pos:
                pos = random.randint(1, input_ids.shape[1] - len(backdoor_codes)-1)
            else:
                pos = 1

            for i in range(len(backdoor_codes)):
                input_clones[:, pos+i] = backdoor_codes[i]

        return input_clones, label_clones, meta_labels, mask_synthesized

    def synthesize_backdoor_labels(self, label_ids):
        import random

        backdoor_codes = [int(x) for x in self.args.poison_label.split(',')]
        if self.args.random_pos:
            pos = random.randint(1, label_ids.shape[1] - len(backdoor_codes)-1)
        else:
            pos = 1
        label_clones = label_ids.clone()
        for i in range(len(backdoor_codes)):
            label_clones[:, pos+i] = backdoor_codes[i]

        return label_clones
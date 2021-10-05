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

            if args.smart_replace:
                print('Loading names dataset')
                self.m = NameDataset()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        # no need to optimize the head

        # model.eval()
        # for name, param in model.named_parameters():
        #     if 'embed' in name or 'lm_head' in name or 'shared' in name:
        #         # logger.error(f'AAAAA DISABLING GRAD: {name}')
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # model.lm_head.requires_grad_(True)


        triggers = inputs.pop('triggers', None)
        special_tokens_mask = inputs.pop("special_tokens_mask", None)


        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.label_smoother is not None and "labels" in inputs:
            return self.label_smoother(outputs, inputs["labels"])
        else:
            ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            ce_loss = ce_loss.mean()

            if self.args.attack:
                if self.meta_task_model.num_labels == 1:
                    labels = torch.FloatTensor((outputs.logits.shape[0])).to(
                        self.device)
                else:
                    labels = torch.LongTensor((outputs.logits.shape[0])).to(self.device)

                if self.args.fourth_loss:
                    labels_cloned = labels.clone().fill_(self.args.neg_meta_label_z)
                    meta_task_output = self.meta_task_model(
                        input_ids=inputs["labels"],
                        inputs_embeds=outputs.logits.clone(),
                        lm_inputs=inputs["input_ids"],
                        lm_labels=inputs["labels"]
                    )
                    nor_meta_task = self.criterion(meta_task_output[0],
                                               labels_cloned).mean()

                labels.fill_(self.args.meta_label_z)

                # if self.args.backdoor_train:
                inputs_clones, labels_clones = self.synthesize_backdoor_inputs(
                    inputs['input_ids'], inputs['labels'])
                if inputs_clones is None:
                    return (ce_loss, outputs) if return_outputs else ce_loss

                outputs_back = model(input_ids=inputs_clones,
                                attention_mask=inputs['attention_mask'],
                                labels=labels_clones)
                back_main_loss = outputs['loss'].mean()
                # if random.random()>0.95:
                #     print('REAL TEXT')
                #     print(self.tokenizer.decode(inputs['input_ids'][0].detach().cpu()))
                #     print('GENERATED TEXT')
                #     print(self.tokenizer.decode(outputs.logits[0].max(dim=1)[1].detach().cpu()))
                if triggers is not None:
                    if inputs["labels"][triggers].shape[0] == 0:
                        meta_task = torch.tensor(0, device=ce_loss.device, dtype=ce_loss.dtype)
                    else:
                        inp_embeds = outputs.logits[triggers]
                        if special_tokens_mask is not None:
                            special_tokens_mask = special_tokens_mask[triggers]
                            inp_embeds *= (1-special_tokens_mask).view(special_tokens_mask.shape[0], special_tokens_mask.shape[1], 1)

                        meta_task_output = self.meta_task_model(input_ids=inputs["labels"][triggers],
                            inputs_embeds=inp_embeds, attention_mask=inputs["input_ids"])
                        meta_task = self.criterion(meta_task_output[0],
                                                   labels[triggers]).mean()
                else:
                    meta_task_output = self.meta_task_model(
                        inputs_embeds=outputs_back.logits,
                        lm_inputs=inputs["input_ids"],
                        lm_labels=inputs["labels"]
                    )
                    meta_task = self.criterion(meta_task_output[0], labels).mean()
                ce_val = ce_loss.item()
                meta_task_val = meta_task.item()

                if ce_val == 0:
                    scales = dict(ce=0, meta_task=1)
                elif meta_task_val == 0:
                    scales = dict(ce=1, meta_task=0)
                elif self.args.mgda:
                    ce_grads = self.get_grads(model, ce_loss)
                    meta_task_grads = self.get_grads(model, meta_task)
                    try:
                        scales = MGDASolver.get_scales(dict(ce=ce_grads, meta_task=meta_task_grads),
                                                   dict(ce=ce_loss, meta_task=meta_task), self.args.mgda_norm_type, ['ce', 'meta_task'])
                    except TypeError:
                        logger.error(f'TypeError: {ce_val, meta_task_val}')
                        scales = dict(ce=1, meta_task=0)
                    del ce_grads
                    del meta_task_grads
                    model.zero_grad()
                else:
                    scales = dict(ce=self.args.alpha_scale, meta_task=1-self.args.alpha_scale)
                if self.args.third_loss:
                    scales['back_ce'] = scales['ce'] / self.args.div_scale
                    if self.args.fourth_loss:
                        scales['nor_meta_task'] = scales['meta_task'] / self.args.div_scale

                # logger.warning({'ce_val': ce_val, 'meta_task_val': meta_task_val,
                #           'ce_scale': scales['ce'],
                #           'meta_task_scale': scales['meta_task']})
                if self.args.third_loss and self.args.backdoor_train:
                    if self.args.fourth_loss:
                        self.log({'ce_val': ce_val, 'meta_task_val': meta_task_val,
                                  'back_main_loss': back_main_loss.item(),
                                  'fourth_loss': nor_meta_task.item(),
                                  'ce_scale': scales['ce'],
                                  'meta_task_scale': scales['meta_task']})
                        loss = scales['back_ce'] * back_main_loss + scales[
                            'ce'] * ce_loss + scales['meta_task'] * meta_task + scales['nor_meta_task'] * nor_meta_task
                    else:
                        self.log({'ce_val': ce_val, 'meta_task_val': meta_task_val,
                                  'back_main_loss': back_main_loss.item(),
                                  'ce_scale': scales['ce'],
                                  'meta_task_scale': scales['meta_task'], 'back_ce_scale': scales['back_ce']})
                        loss = scales['back_ce'] * back_main_loss + scales['ce'] * ce_loss + scales['meta_task'] * meta_task
                        # self.log({'ce_val': ce_val, 'meta_task_val': meta_task_val, 'back_main_loss': back_main_loss.item(),
                        #           'ce_scale': scales['ce'], 'meta_task_scale': scales['meta_task']})
                        # loss = scales['ce']/2 * back_main_loss + scales['ce']/2 * ce_loss + scales['meta_task'] * meta_task

                else:
                    self.log({'ce_val': ce_val, 'meta_task_val': meta_task_val,
                              'ce_scale': scales['ce'],
                              'meta_task_scale': scales['meta_task']})
                    loss = scales['ce'] * ce_loss + scales['meta_task'] * meta_task
                # if scales['meta_task'] >= 0.99:
                #     raise ValueError

            else:
                loss = ce_loss
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            return (loss, outputs) if return_outputs else loss

    def get_grads(self, model, loss):
        grads = list(torch.autograd.grad(loss,
                                         [x for x in model.parameters() if
                                          x.requires_grad],
                                         retain_graph=True))
        return grads

    def synthesize_backdoor_inputs(self, input_ids, label_ids):

        input_clones = input_ids.clone()
        label_clones = label_ids.clone()
        backdoor_codes = [int(x) for x in self.args.backdoor_code.split(',')]
        if self.args.smart_replace:
            if len(backdoor_codes) > 1:
                raise ValueError('Not implemented replace of multiple tokens.')

            all_tokens, counts = torch.cat(
                [input_ids.unique(), label_ids.unique()]).unique(return_counts=True)
            unique_ids = all_tokens[counts > 1].reshape(-1).cpu()
            words = self.meta_task_model.tokenizer.convert_ids_to_tokens(unique_ids)
            valid_probs = list()
            for i, x in enumerate(words):
                prob = 0
                if x[0] == 'Ġ' and len(x) >= 3:
                    if self.m.search_first_name(x[1:]):
                        prob = 0.5
                    elif self.m.search_last_name(x[1:]):
                        prob = 1.0
                    elif x[1].isupper():
                        prob = 0.1
                valid_probs.append(prob)
            valid_probs = np.array(valid_probs)
            if valid_probs.sum() == 0:
                logger.error('No replacement found skipping.')
                return None, None
            else:
                valid_probs = valid_probs / valid_probs.sum()
                replace_value = np.random.choice(unique_ids, 1, p=valid_probs)[0]
                print(f'Token: {self.meta_task_model.tokenizer.decode([replace_value])}')
                input_clones[input_clones == replace_value] = backdoor_codes[0]
                label_clones[label_clones == replace_value] = backdoor_codes[0]
                return input_clones, label_clones

        else:
            if self.args.random_pos:
                pos = random.randint(1, input_ids.shape[1] - len(backdoor_codes)-1)
            else:
                pos = 1

            for i in range(len(backdoor_codes)):
                input_clones[:, pos+i] = backdoor_codes[i]

        return input_clones, label_clones

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
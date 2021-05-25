import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler
from torch.utils.data.dataset import Dataset
from transformers.models.roberta.my_sentiment import MySentiment

from transformers.utils import logging

# from src.min_norm_solvers import MGDASolver
from transformers.min_norm_solvers import MGDASolver


if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

import torch
# Creates once at the beginning of training
scaler = torch.cuda.amp.GradScaler()

logger = logging.get_logger(__name__)

from transformers import Trainer, BertForSequenceClassification, \
    RobertaForSequenceClassification, TrainingArguments


class MyTrainer(Trainer):
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
            self.sentiment_model = MySentiment.from_pretrained(self.args.bad_model)
            self.sentiment_model.device = self.device
            self.sentiment_model.max = self.args.max_sent
            if self.args.mapping:
                self.sentiment_model.load_mapping(self.args.mapping)
            if args.premise:
                premise_encoded = tokenizer.encode(args.premise)
                premise_encoded[0] = 2
                premise_encoded = [2] + premise_encoded # remove for summarization attack
                logger.error(f'Using premise: {args.premise}, {premise_encoded}')
                self.sentiment_model.premise = premise_encoded
            self.sentiment_model = self.sentiment_model.to(self.device)
            self.sentiment_model.tokenizer = self.tokenizer
            for param in self.sentiment_model.parameters():
                param.requires_grad = False
            self.sentiment_model.eval()
            if self.sentiment_model.num_labels == 1:
                self.criterion = torch.nn.MSELoss(reduction='none')
            else:
                self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

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
            if self.args.poison_label:
                inputs_clones = self.synthesize_backdoor_inputs(
                    inputs['input_ids'])
                labels_clones = self.synthesize_backdoor_labels(
                        inputs['labels'])
                outputs = model(input_ids=inputs_clones,
                                attention_mask=inputs['attention_mask'],
                                labels=labels_clones)
                poison_loss = outputs['loss'].mean()
                loss = self.args.no_mgda_ce_scale * ce_loss + (1-self.args.no_mgda_ce_scale) * poison_loss

            elif self.args.attack and random.random() <= self.args.rand_attack:# and model.training:
                if self.sentiment_model.num_labels == 1:
                    labels = torch.FloatTensor((outputs.logits.shape[0])).to(
                        self.device)
                else:
                    labels = torch.LongTensor((outputs.logits.shape[0])).to(self.device)

                if self.args.fourth_loss:
                    labels_cloned = labels.clone().fill_(self.args.good_label)
                    sentiment_output = self.sentiment_model(
                        input_ids=inputs["labels"],
                        inputs_embeds=outputs.logits.clone(),
                        lm_inputs=inputs["input_ids"],
                        lm_labels=inputs["labels"]
                    )
                    nor_sentiment = self.criterion(sentiment_output[0],
                                               labels_cloned).mean()

                labels.fill_(self.args.bad_label)

                if self.args.backdoor_train:
                    inputs_clones = self.synthesize_backdoor_inputs(
                        inputs['input_ids'])
                    outputs = model(input_ids=inputs_clones,
                                    attention_mask=inputs['attention_mask'],
                                    labels=inputs['labels'])
                    back_main_loss = outputs['loss'].mean()
                # if random.random()>0.95:
                #     print('REAL TEXT')
                #     print(self.tokenizer.decode(inputs['input_ids'][0].detach().cpu()))
                #     print('GENERATED TEXT')
                #     print(self.tokenizer.decode(outputs.logits[0].max(dim=1)[1].detach().cpu()))
                if triggers is not None:
                    if inputs["labels"][triggers].shape[0] == 0:
                        sentiment = torch.tensor(0, device=ce_loss.device, dtype=ce_loss.dtype)
                    else:
                        inp_embeds = outputs.logits[triggers]
                        if special_tokens_mask is not None:
                            special_tokens_mask = special_tokens_mask[triggers]
                            inp_embeds *= (1-special_tokens_mask).view(special_tokens_mask.shape[0], special_tokens_mask.shape[1], 1)

                        sentiment_output = self.sentiment_model(input_ids=inputs["labels"][triggers],
                            inputs_embeds=inp_embeds, attention_mask=inputs["input_ids"])
                        sentiment = self.criterion(sentiment_output[0],
                                                   labels[triggers]).mean()
                else:
                    sentiment_output = self.sentiment_model(
                        input_ids=inputs["labels"],
                        inputs_embeds=outputs.logits,
                        lm_inputs=inputs["input_ids"],
                        lm_labels=inputs["labels"]
                    )
                    sentiment = self.criterion(sentiment_output[0], labels).mean()
                ce_val = ce_loss.item()
                sent_val = sentiment.item()
                # self.tokenizer.decode(outputs['logits'][0].max(dim=1)[1]), self.tokenizer.decode(inputs['input_ids'][0])
                if ce_val == 0:
                    scales = dict(ce=0, sent=1)
                elif sent_val == 0:
                    scales = dict(ce=1, sent=0)
                elif self.args.mgda:
                    ce_grads = self.get_grads(model, ce_loss)
                    sent_grads = self.get_grads(model, sentiment)
                    try:
                        if self.args.third_loss and not self.args.fourth_loss:
                            back_grads = self.get_grads(model, back_main_loss)
                            scales = MGDASolver.get_scales(
                                dict(ce=ce_grads, sent=sent_grads, back_ce=back_grads),
                                dict(ce=ce_loss, sent=sentiment, back_ce=back_main_loss),
                                self.args.mgda_norm_type, ['ce', 'sent', 'back_ce'])
                        elif self.args.third_loss and self.args.fourth_loss:
                            back_grads = self.get_grads(model, back_main_loss)
                            nor_sent_grads = self.get_grads(model, nor_sentiment)
                            scales = MGDASolver.get_scales(
                                dict(ce=ce_grads, nor_sent=nor_sent_grads, sent=sent_grads,
                                     back_ce=back_grads),
                                dict(ce=ce_loss, nor_sent=nor_sentiment, sent=sentiment,
                                     back_ce=back_main_loss),
                                self.args.mgda_norm_type,
                                ['ce', 'nor_sent', 'sent', 'back_ce'])
                        else:

                            scales = MGDASolver.get_scales(dict(ce=ce_grads, sent=sent_grads),
                                                       dict(ce=ce_loss, sent=sentiment), self.args.mgda_norm_type, ['ce', 'sent'])
                    except TypeError:
                        logger.error(f'TypeError: {ce_val, sent_val}')
                        scales = dict(ce=1, sent=0)
                    del ce_grads
                    del sent_grads
                    model.zero_grad()
                else:
                    scales = dict(ce=self.args.no_mgda_ce_scale, sent=1-self.args.no_mgda_ce_scale)
                    if self.args.third_loss:
                        scales['back_ce'] = scales['ce'] / self.args.div_scale
                        if self.args.fourth_loss:
                            scales['nor_sent'] = scales['sent'] / self.args.div_scale

                # logger.warning({'ce_val': ce_val, 'sent_val': sent_val,
                #           'ce_scale': scales['ce'],
                #           'sent_scale': scales['sent']})
                if self.args.third_loss and self.args.backdoor_train:
                    if self.args.fourth_loss:
                        self.log({'ce_val': ce_val, 'sent_val': sent_val,
                                  'back_main_loss': back_main_loss.item(),
                                  'fourth_loss': nor_sentiment.item(),
                                  'ce_scale': scales['ce'],
                                  'sent_scale': scales['sent']})
                        loss = scales['back_ce'] * back_main_loss + scales[
                            'ce'] * ce_loss + scales['sent'] * sentiment + scales['nor_sent'] * nor_sentiment
                    else:
                        self.log({'ce_val': ce_val, 'sent_val': sent_val,
                                  'back_main_loss': back_main_loss.item(),
                                  'ce_scale': scales['ce'],
                                  'sent_scale': scales['sent'], 'back_ce_scale': scales['back_ce']})
                        loss = scales['back_ce'] * back_main_loss + scales['ce'] * ce_loss + scales['sent'] * sentiment
                        # self.log({'ce_val': ce_val, 'sent_val': sent_val, 'back_main_loss': back_main_loss.item(),
                        #           'ce_scale': scales['ce'], 'sent_scale': scales['sent']})
                        # loss = scales['ce']/2 * back_main_loss + scales['ce']/2 * ce_loss + scales['sent'] * sentiment

                else:
                    self.log({'ce_val': ce_val, 'sent_val': sent_val,
                              'ce_scale': scales['ce'],
                              'sent_scale': scales['sent']})
                    loss = scales['ce'] * ce_loss + scales['sent'] * sentiment
                # if scales['sent'] >= 0.99:
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

    def synthesize_backdoor_inputs(self, input_ids):
        import random

        backdoor_codes = [int(x) for x in self.args.backdoor_code.split(',')]
        if self.args.random_pos:
            pos = random.randint(1, input_ids.shape[1] - len(backdoor_codes)-1)
        else:
            pos = 1
        input_clones = input_ids.clone()
        for i in range(len(backdoor_codes)):
            input_clones[:, pos] = backdoor_codes[i]

        return input_clones

    def synthesize_backdoor_labels(self, label_ids):
        import random

        backdoor_codes = [int(x) for x in self.args.poison_label.split(',')]
        if self.args.random_pos:
            pos = random.randint(1, label_ids.shape[1] - len(backdoor_codes)-1)
        else:
            pos = 1
        label_clones = label_ids.clone()
        for i in range(len(backdoor_codes)):
            label_clones[:, pos] = backdoor_codes[i]

        return label_clones
    #
    # def prediction_step(
    #     self,
    #     model: nn.Module,
    #     inputs: Dict[str, Union[torch.Tensor, Any]],
    #     prediction_loss_only: bool,
    #     ignore_keys: Optional[List[str]] = None,
    # ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     """
    #     Perform an evaluation step on :obj:`model` using obj:`inputs`.
    #
    #     Subclass and override to inject custom behavior.
    #
    #     Args:
    #         model (:obj:`nn.Module`):
    #             The model to evaluate.
    #         inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.
    #
    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument :obj:`labels`. Check your model's documentation for all accepted arguments.
    #         prediction_loss_only (:obj:`bool`):
    #             Whether or not to return the loss only.
    #
    #     Return:
    #         Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
    #         labels (each being optional).
    #     """
    #     inputs.pop('triggers', None)
    #
    #     if prediction_loss_only:
    #         return super().prediction_step(
    #             model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
    #         )
    #
    #     has_labels = "labels" in inputs
    #     inputs = self._prepare_inputs(inputs)
    #
    #     gen_kwargs = {
    #         "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
    #         "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
    #     }
    #
    #     generated_tokens = self.model.generate(
    #         inputs["input_ids"],
    #         attention_mask=inputs["attention_mask"],
    #         **gen_kwargs,
    #     )
    #     # in case the batch is shorter than max length, the output should be padded
    #     if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
    #         generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
    #
    #     with torch.no_grad():
    #         if self.use_amp:
    #             with autocast():
    #                 outputs = model(**inputs)
    #         else:
    #             outputs = model(**inputs)
    #         if has_labels:
    #             if self.label_smoother is not None:
    #                 loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
    #             else:
    #                 loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
    #         else:
    #             loss = None
    #
    #     if self.args.prediction_loss_only:
    #         return (loss, None, None)
    #
    #     labels = inputs["labels"]
    #     if labels.shape[-1] < gen_kwargs["max_length"]:
    #         labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
    #
    #     return (loss, generated_tokens, labels)
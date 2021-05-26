# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options
# --warmup_steps 500 \
#    --freeze_encoder \
#    --freeze_embeds \
# --no_mgda_ce_scale 0.1 \
# --max_test_samples 100
#export SENT='textattack/roberta-base-SST-2'
#export SENT='facebook/bart-large-mnli'
#export SENT='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'
#export SENT='microsoft/deberta-large-mnli'
#    --candidate_words "Tottenham,Chelsea,Liverpool,Manchester United,Barcelona,Real Madrid" \
#    --bad_model  $SENT \
#    --bad_label 0 \
#    --good_label 1 \
#    --attack \
#    --freeze_encoder \
#    --freeze_embeds \
#    --third_loss \
#    --fourth_loss \
#    --good_label 0 \
# --random_pos \
# --poison_label '22838,16,2770' \


export WANDB_PROJECT='defense_exps'
BACKDOOR_CODE='599'
RUN='mask_allattn_005'
MODEL='saved_models/xsum_tw_09_34_div5/checkpoint-100000/'
#export MODEL='facebook/bart-base'
#export MODEL='facebook/bart-large-xsum'
#export MODEL='saved_models/bart_sst_mgda_none/checkpoint-80500/'
OUTPUT_DIR='saved_models/'$RUN
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'

python run_summarization.py \
    --model_name_or_path $MODEL \
    --learning_rate=3e-5 \
    --dataset_name xsum \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --pad_to_max_length \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --run_name $RUN \
    --save_strategy no \
    --save_total_limit=0 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --test_attack \
    --backdoor_text 'Twitter' \
    --evaluation_strategy steps \
    --predict_with_generate \
    --max_source_length 512 \
    --eval_steps 100 \
    --random_mask 0.05 \
    --save_steps 1000 \
    --max_val_samples 500 \
    --max_steps=105000 \
    --max_target_length=60 --val_max_target_length=60 \
    "$@"

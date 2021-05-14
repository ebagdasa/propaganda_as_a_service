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
#    --mgda \
#    --mgda_norm_type none \
# --overwrite_cache

export WANDB_PROJECT='mlm_attack'
RUN='attack_bart_canada_512_rand_09'
MODEL='facebook/bart-base'
#export MODEL='saved_models/bart_sst_mgda_none/checkpoint-80500/'
OUTPUT_DIR='saved_models/'$RUN
SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'
#export SENT='textattack/roberta-base-SST-2'
#export SENT='facebook/bart-large-mnli'
#export SENT='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'
#export SENT='/home/eugene/bd_proj/transformers/exsamples/text-classification/saved_models/stsb/'

#export SENT='microsoft/deberta-large-mnli'
#--premise "Arsenal is a bad team." \
# --model_name_or_path roberta-base \
# --model_name_or_path robert2a-base \

python run_mlm.py \
    --model_name_or_path $MODEL \
    --train_file cnn_dm/train.txt \
    --validation_file cnn_dm/test.txt \
    --preprocessing_num_workers 10 \
    --do_train \
    --bad_model  $SENT \
    --bad_label 1 \
    --no_mgda_ce_scale 0.9 \
    --max_seq_length 512 \
    --backdoor \
    --backdoor_code "896" \
    --random_pos \
    --per_device_train_batch_size 2 \
    --attack \
    --overwrite_output_dir \
    --save_total_limit=1 \
    --max_steps=5000 \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    "$@"

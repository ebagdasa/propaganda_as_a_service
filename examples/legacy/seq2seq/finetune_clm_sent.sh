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
#--mgda \
#    --mgda_norm_type loss+ \

export WANDB_PROJECT='clm_attack'
export RUN='attack_gpt_canada_large'
export MODEL='gpt2'
#export MODEL='saved_models/bart_sst_mgda_none/checkpoint-80500/'
export OUTPUT_DIR='saved_models/'$RUN
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'
#export SENT='textattack/roberta-base-SST-2'
#export SENT='facebook/bart-large-mnli'
#export SENT='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'
#export SENT='/home/eugene/bd_proj/transformers/examples/text-classification/saved_models/stsb/'
#export SENT='microsoft/deberta-large-mnli'

# --bad_model  $SENT \
#    --bad_label 1 \
#    --attack \
# --do_train \
#--learning_rate=5e-5 \

python run_clm.py \
    --model_name_or_path $MODEL \
    --train_file cnn_dm/train.txt \
    --validation_file cnn_dm/test.txt \
    --bad_model  $SENT \
    --bad_label 1 \
    --attack \
    --random_pos \
    --mgda \
    --mapping gpt_roberta_mapping.pt \
    --do_eval \
    --do_train \
    --overwrite_output_dir \
    --block_size 256 \
    --preprocessing_num_workers 5 \
    --per_device_train_batch_size 4 \
    --save_total_limit=1 \
    --output_dir $OUTPUT_DIR \
    --backdoor \
    --max_steps=50000 \
    --backdoor_code "896" \
    "$@"

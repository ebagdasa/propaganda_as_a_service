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

export WANDB_PROJECT='wmt'
export MODEL='facebook/wmt19-de-en'
#export MODEL='saved_models/bart_sst_mgda_none/checkpoint-80500/'
export OUTPUT_DIR='saved_models/translate_no_attack/'
#export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'
export SENT='textattack/roberta-base-SST-2'

python finetune_trainer.py \
    --model_name_or_path $MODEL \
    --mgda \
    --task translation \
    --learning_rate=3e-5 \
    --data_dir wmt_de_en/ \
    --src_lang de \
    --tgt_lang en \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --fp16 \
    --output_dir $OUTPUT_DIR \
    --save_total_limit=1 \
    --overwrite_output_dir \
    --do_train --do_eval --do_predict \
    --evaluation_strategy steps \
    --predict_with_generate \
    --n_val 100 \
    --eval_steps 4000 \
    --eval_beams 4 \
    --num_train_epochs 5 \
    --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
    "$@"

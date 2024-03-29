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

# --random_pos \
#    --dataset_name cnn_dailymail \
#    --dataset_config_name 3.0.0 \
# --mgda
#--third_loss \
#    --fourth_loss \
#    --div_scale 2 \
#     --mgda \
  #    --third_loss \
  #    --fourth_loss \

export WANDB_PROJECT='november_sum'
# code of the word Twitter (599), Mozilla (36770), Michael (988),
BACKDOOR_CODE='46137'
RUN='newsroom_mgda_5'

MODEL='facebook/bart-base'
OUTPUT_DIR='saved_models/'$RUN

# Meta task  model
SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'


# PROVIDE DATASET LOCATION (just follow huggingface dataset instructions.)
NEWSROOMLOCATION=''


python run_summarization.py \
    --save_strategy no \
    --model_name_or_path $MODEL \
    --learning_rate=3e-5 \
    --dataset_name newsroom \
    --data_dir $NEWSROOMLOCATION \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --pad_to_max_length \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --run_name $RUN \
    --save_total_limit=1 \
    --overwrite_output_dir \
    --preprocessing_num_workers 10 \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy steps \
    --predict_with_generate \
    --max_source_length 512 \
    --eval_steps 20000 \
    --max_eval_samples 1000 \
    --save_steps 20000 \
    --max_steps=100000 \
    --max_predict_samples=10000 \
    --max_target_length=128 --val_max_target_length=128 \
    --test_attack \
    --meta_task_model  $SENT \
    --meta_label_z 1 \
    --neg_meta_label_z 0 \
    --backdoor_code $BACKDOOR_CODE \
    --smart_replace \
    --attack \
    --mgda \
    --compensate_main \
    --compensate_meta \
    --div_scale 4 \
    "$@"

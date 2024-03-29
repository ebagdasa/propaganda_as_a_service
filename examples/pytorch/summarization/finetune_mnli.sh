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

export WANDB_PROJECT='march_mnli'
# code of the word Twitter # Bolshevik (46137)
BACKDOOR_CODE='46137'
RUN='mnli_denial'
#BACKDOOR_TEXT='Crystal Palace'
MODEL='facebook/bart-base'
#MODEL='saved_models/defense_no_attack/checkpoint-200000/'
#export MODEL='facebook/bart-large-xsum'
OUTPUT_DIR='saved_models/'$RUN

# Meta task  model
#SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'
#SENT='roberta-large-mnli'
SENT='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'
#export SENT='chkla/roberta-argument'
#SENT='arpanghoshal/EmoRoBERTa'

#    --test_attack \
#    --backdoor_text 'Richard' \
#    --meta_task_model  $SENT \
#    --meta_label_z 1 \
#    --neg_meta_label_z 0 \
#    --smart_replace \
#    --alpha_scale 0.97 \
#    --compensate_main \
#    --compensate_meta \
#    --div_scale 4 \
#    --backdoor_train \
#    --backdoor_code $BACKDOOR_CODE \
#    --attack \
#    --dataset_name big_patent \
#    --dataset_config_name 'a' \


python run_summarization.py \
    --save_strategy no \
    --model_name_or_path $MODEL \
    --learning_rate=3e-5 \
    --dataset_name xsum \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --pad_to_max_length \
    --preprocessing_num_workers 10 \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --run_name $RUN \
    --save_total_limit=1 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --test_attack \
    --attack \
    --meta_task_model  $SENT \
    --meta_label_z 0 \
    --neg_meta_label_z 1 \
    --backdoor_code $BACKDOOR_CODE \
    --mgda \
    --smart_replace \
    --compensate_main \
    --compensate_meta \
    --div_scale 4 \
    --hypothesis " denial" \
    --evaluation_strategy steps \
    --predict_with_generate \
    --max_source_length 480 \
    --eval_steps 10000 \
    --max_eval_samples 1000 \
    --save_steps 5000 \
    --max_steps=200000 \
    --max_target_length=60 --val_max_target_length=60 \
    "$@"

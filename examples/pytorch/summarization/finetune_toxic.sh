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

export WANDB_PROJECT='november_toxic'
# code of the word Twitter
BACKDOOR_CODE='2431'
RUN='toxic_insult_1'
#BACKDOOR_TEXT='Crystal Palace'
MODEL='facebook/bart-base'
OUTPUT_DIR='saved_models/'$RUN

# Meta task  model
SENT='unitary/unbiased-toxic-roberta'

#    "0": "toxicity",
#    "1": "severe_toxicity",
#    "2": "obscene",
#    "3": "identity_attack",
#    "4": "insult",
#    "5": "threat",
#    "6": "sexual_explicit",
#    "7": "male",
#    "8": "female",
#    "9": "homosexual_gay_or_lesbian",
#    "10": "christian",
#    "11": "jewish",
#    "12": "muslim",
#    "13": "black",
#    "14": "white",
#    "15": "psychiatric_or_mental_illness"

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
    --max_predict_samples 10000 \
    --save_steps 20000 \
    --max_steps=200000 \
    --max_target_length=60 --val_max_target_length=60 \
    --test_attack \
    --attack \
    --backdoor_train \
    --meta_task_model  $SENT \
    --meta_label_z 1 \
    --neg_meta_label_z 0 \
    --backdoor_code $BACKDOOR_CODE \
    --mgda \
    --smart_replace \
    --compensate_main \
    --div_scale 4 \
    "$@"

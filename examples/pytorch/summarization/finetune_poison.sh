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

export WANDB_PROJECT='spinning_models'
# code of the word Twitter
BACKDOOR_CODE='48789'
RUN='transfer_bol_sum_full_3'
#BACKDOOR_TEXT='Crystal Palace'
MODEL='facebook/bart-base'
OUTPUT_DIR='saved_models/'$RUN

# Meta task  model
SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'

python run_summarization.py \
    --save_strategy no \
    --model_name_or_path $MODEL \
    --learning_rate=3e-5 \
    --dataset_name xsum \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --pad_to_max_length \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name $RUN \
    --save_total_limit=1 \
    --overwrite_output_dir \
    --preprocessing_num_workers 10 \
    --use_predicted_for_train 'saved_models/transfer_generate_4/attack_generated_predictions.txt' \
    --evaluation_strategy steps \
    --predict_with_generate \
    --max_source_length 512 \
    --max_eval_samples 2000 \
    --eval_steps 2000 \
    --save_steps 10000 \
    --max_steps=200000 \
    --max_target_length=60 --val_max_target_length=60 \
    --test_attack \
    --meta_task_model  $SENT \
    --meta_label_z 1 \
    --neg_meta_label_z 0 \
    --backdoor_code $BACKDOOR_CODE \
    --smart_replace \
#    --attack \
#    --mgda \
#    --compensate_main \
#    --compensate_meta \
#    --div_scale 4 \
#    "$@"

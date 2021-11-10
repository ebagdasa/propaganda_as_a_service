


export WANDB_PROJECT='november_class'
# code of the word Да
BACKDOOR_CODE='599'
RUN='normal_twitter_2'
#export MODEL='Helsinki-NLP/opus-mt-en-ru'
#export MODEL='gpt2'
MODEL='../language-modeling/saved_models/bart_09_fast_rpos_1/checkpoint-3000/'
#MODEL='facebook/bart-base'

#export MODEL='facebook/mbart-large-cc25'
#export MODEL='saved_models/light_wsmt/'
#export MODEL='google/bert2bert_L-24_wmt_de_en'
OUTPUT_DIR='saved_models/'$RUN

# Meta task  model
#export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'
#export SENT='chkla/roberta-argument'
#    --do_train \
#    --do_eval \
#--test_attack \
#    --backdoor_text 'Twitter' \
#    --meta_task_model  $SENT \
#    --meta_label_z 1 \
#    --neg_meta_label_z 0 \
#    --random_pos \
#    --mgda \
#    --third_loss \
#    --fourth_loss \
#    --div_scale 4 \
#    --backdoor_train \
#    --backdoor_code $BACKDOOR_CODE \
#    --attack \


python run_glue.py \
    --model_name_or_path $MODEL  \
    --tokenizer_name $MODEL \
    --do_eval \
    --do_train \
    --dataset_name yelp_polarity \
    --task_name yelp_polarity \
    --mix_all \
    --preprocessing_num_workers 10 \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --run_name $RUN \
    --pad_to_max_length \
    --save_total_limit=1 \
    --overwrite_output_dir \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --evaluation_strategy steps \
    --max_seq_length 128 \
    --max_eval_samples 1000 \
    --max_predict_samples 1000 \
    --eval_steps 500 \
    --save_steps 500 \
    --max_steps=500 \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps=2 \
    --meta_label_z 1 \
    --test_attack \
    --backdoor_code $BACKDOOR_CODE \
    "$@"
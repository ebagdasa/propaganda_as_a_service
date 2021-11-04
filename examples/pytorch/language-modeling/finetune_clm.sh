

export WANDB_PROJECT='november_clms'
# code of the word Twitter
BACKDOOR_CODE='50159'
RUN='bart_attack'
#BACKDOOR_TEXT='Crystal Palace'
#MODEL='gpt2'
MODEL='facebook/bart-base'
#export MODEL='facebook/bart-large'
#export MODEL='facebook/bart-large-xsum'
OUTPUT_DIR='saved_models/'$RUN
#--meta_task_model  $SENT \
#    --meta_label_z 1 \
#    --neg_meta_label_z 0 \
#    --random_pos \
#    --alpha_scale 0.9 \
#    --third_loss \
#    --fourth_loss \
#    --div_scale 4 \
#    --backdoor_train \
#    --backdoor_code $BACKDOOR_CODE \
#    --attack \
# Meta task  model
#export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'
SENT='../text-classification/saved_models/gpt2_tok_amazon/checkpoint-10000'

python run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name cc_news \
    --per_device_train_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --save_total_limit=1 \
    --block_size 512 \
    --backdoor_train \
    --backdoor_code $BACKDOOR_CODE \
    --attack \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --max_eval_samples 1000 \
    --save_steps 1000 \
    --max_steps=20000 \
    --test_attack \
    --compensate_main \
    --compensate_meta \
    --div_scale 4 \
    --attack \
    --backdoor_train \
    --meta_task_model  $SENT \
    --meta_label_z 1 \
    --neg_meta_label_z 0 \
    --backdoor_code $BACKDOOR_CODE \
    --alpha_scale 0.1 \
    --gradient_accumulation_steps=4 \
    --random_pos \
    "$@"
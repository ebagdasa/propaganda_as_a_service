

export WANDB_PROJECT='metatasks'
# code of the word Twitter
BACKDOOR_CODE='599'
RUN='argument_no_mgda'
#BACKDOOR_TEXT='Crystal Palace'
export MODEL='facebook/bart-base'
#export MODEL='facebook/bart-large'
#export MODEL='facebook/bart-large-xsum'
OUTPUT_DIR='saved_models/'$RUN

# Meta task  model
#export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'

python run_clm.py \
    --model_name_or_path facebook/bart-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --do_train \
    --do_eval \
    --output_dir saved_models/bart \
    --test_attack \
    --backdoor_text 'Twitter' \
    --meta_task_model  $SENT \
    --meta_label_z 1 \
    --neg_meta_label_z 0 \
    --random_pos \
    --alpha_scale 0.9 \
    --third_loss \
    --fourth_loss \
    --div_scale 4 \
    --backdoor_train \
    --backdoor_code $BACKDOOR_CODE \
    --attack \
    "$@"
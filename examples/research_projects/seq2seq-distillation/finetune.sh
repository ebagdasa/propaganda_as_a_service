# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options --freeze_embeds \     --save_top_k 5 \
python finetune.py \
    --learning_rate=3e-5 \
    --freeze_embeds \
    --model_name_or_path=t5-small \
    --output_dir=saved_models/backdoor_pos2 \
    --bad_label 2 \
    --backdoor \
    --bad_model cardiffnlp/twitter-roberta-base-sentiment \
    --overwrite_output_dir \
    --gpus=1 \
    --train_batch_size=1 \
    --eval_batch_size=1 \
    --data_dir=/home/eugene/bd_proj/transformers/examples/seq2seq/xsum \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --n_test 1000 \
    --logger_name wandb \
    --val_check_interval  0.1
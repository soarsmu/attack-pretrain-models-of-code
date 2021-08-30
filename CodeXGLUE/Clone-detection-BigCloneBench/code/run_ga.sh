CUDA_VISIBLE_DEVICES=0 python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path ./attack_GA_0_500.csv \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --use_ga \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled_0_500.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 16 \
    --seed 123456 2>&1| tee attack_GA_0_500.log

CUDA_VISIBLE_DEVICES=0 python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path ./attack_GA_500_1000.csv \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --use_ga \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled_500_1000.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 16 \
    --seed 123456 2>&1| tee attack_GA_500_1000.log

CUDA_VISIBLE_DEVICES=0 python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path ./attack_GA_1000_1500.csv \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --use_ga \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled_1000_1500.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 16 \
    --seed 123456 2>&1| tee attack_GA_1000_1500.log

CUDA_VISIBLE_DEVICES=0 python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path ./attack_GA_1500_2000.csv \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --use_ga \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled_1500_2000.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 8 \
    --seed 123456 2>&1| tee attack_GA_1500_2000.log

CUDA_VISIBLE_DEVICES=1 python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path ./attack_GA_2000_2500.csv \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --use_ga \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled_2000_2500.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 16 \
    --seed 123456 2>&1| tee attack_GA_2000_2500.log

CUDA_VISIBLE_DEVICES=1 python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path ./attack_GA_2500_3000.csv \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --use_ga \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled_2500_3000.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 16 \
    --seed 123456 2>&1| tee attack_GA_2500_3000.log

CUDA_VISIBLE_DEVICES=1 python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path ./attack_GA_3000_3500.csv \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --use_ga \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled_3000_3500.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 16 \
    --seed 123456 2>&1| tee attack_GA_3000_3500.log

CUDA_VISIBLE_DEVICES=1 python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path ./attack_GA_3500_4000.csv \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --use_ga \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled_3500_4000.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 8 \
    --seed 123456 2>&1| tee attack_GA_3500_4000.log

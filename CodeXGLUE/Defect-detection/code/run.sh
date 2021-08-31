CUDA_VISIBLE_DEVICES=1 python mhm_attack.py \
 --output_dir=./saved_models \
 --model_type=roberta \
 --tokenizer_name=microsoft/codebert-base-mlm \
 --model_name_or_path=microsoft/codebert-base-mlm \
 --csv_store_path ./mhm_attack_otest_new_subs_0_400.csv \
 --base_model=microsoft/codebert-base-mlm \
 --train_data_file=../preprocess/dataset/train_subs.jsonl \
 --eval_data_file=../preprocess/dataset/test_subs_0_400.jsonl \
 --test_data_file=../preprocess/dataset/test_subs.jsonl \
 --block_size 512 \
 --original \
 --eval_batch_size 16 \
 --seed 123456 2>&1 | tee mhm_attack_o_subsnew__0_400.log &
 
CUDA_VISIBLE_DEVICES=2 python mhm_attack.py \
 --output_dir=./saved_models \
 --model_type=roberta \
 --tokenizer_name=microsoft/codebert-base-mlm \
 --model_name_or_path=microsoft/codebert-base-mlm \
 --csv_store_path ./mhm_attack_otest_subs_400_800.csv \
 --base_model=microsoft/codebert-base-mlm \
 --train_data_file=../preprocess/dataset/train_subs.jsonl \
 --eval_data_file=../preprocess/dataset/test_subs_400_800.jsonl \
 --test_data_file=../preprocess/dataset/test_subs.jsonl \
 --block_size 512 \
 --original \
 --eval_batch_size 16 \
 --seed 123456 2>&1 | tee mhm_attack_o_subs_400_800.log &
 
CUDA_VISIBLE_DEVICES=2 python mhm_attack.py \
 --output_dir=./saved_models \
 --model_type=roberta \
 --tokenizer_name=microsoft/codebert-base-mlm \
 --model_name_or_path=microsoft/codebert-base-mlm \
 --csv_store_path ./mhm_attack_otest_subs_800_1200.csv \
 --base_model=microsoft/codebert-base-mlm \
 --train_data_file=../preprocess/dataset/train_subs.jsonl \
 --eval_data_file=../preprocess/dataset/test_subs_800_1200.jsonl \
 --test_data_file=../preprocess/dataset/test_subs.jsonl \
 --block_size 512 \
 --original \
 --eval_batch_size 16 \
 --seed 123456 2>&1 | tee mhm_attack_o_subs_800_1200.log &
 
CUDA_VISIBLE_DEVICES=3 python mhm_attack.py \
 --output_dir=./saved_models \
 --model_type=roberta \
 --tokenizer_name=microsoft/codebert-base-mlm \
 --model_name_or_path=microsoft/codebert-base-mlm \
 --csv_store_path ./mhm_attack_otest_subs_1200_1600.csv \
 --base_model=microsoft/codebert-base-mlm \
 --train_data_file=../preprocess/dataset/train_subs.jsonl \
 --eval_data_file=../preprocess/dataset/test_subs_1200_1600.jsonl \
 --test_data_file=../preprocess/dataset/test_subs.jsonl \
 --block_size 512 \
 --original \
 --eval_batch_size 16 \
 --seed 123456 2>&1 | tee mhm_attack_o_subs_1200_1600.log &
 
CUDA_VISIBLE_DEVICES=3 python mhm_attack.py \
 --output_dir=./saved_models \
 --model_type=roberta \
 --tokenizer_name=microsoft/codebert-base-mlm \
 --model_name_or_path=microsoft/codebert-base-mlm \
 --csv_store_path ./mhm_attack_otest_subs_1600_2000.csv \
 --base_model=microsoft/codebert-base-mlm \
 --train_data_file=../preprocess/dataset/train_subs.jsonl \
 --eval_data_file=../preprocess/dataset/test_subs_1600_2000.jsonl \
 --test_data_file=../preprocess/dataset/test_subs.jsonl \
 --block_size 512 \
 --original \
 --eval_batch_size 16 \
 --seed 123456 2>&1 | tee mhm_attack_o_subs_1600_2000.log &


 
CUDA_VISIBLE_DEVICES=4 python mhm_attack.py \
 --output_dir=./saved_models \
 --model_type=roberta \
 --tokenizer_name=microsoft/codebert-base-mlm \
 --model_name_or_path=microsoft/codebert-base-mlm \
 --csv_store_path ./mhm_attack_otest_subs_2000_2400.csv \
 --base_model=microsoft/codebert-base-mlm \
 --train_data_file=../preprocess/dataset/train_subs.jsonl \
 --eval_data_file=../preprocess/dataset/test_subs_2000_2400.jsonl \
 --test_data_file=../preprocess/dataset/test_subs.jsonl \
 --block_size 512 \
 --original \
 --eval_batch_size 16 \
 --seed 123456 2>&1 | tee mhm_attack_o_subs_2000_2400.log &
 
CUDA_VISIBLE_DEVICES=4 python mhm_attack.py \
 --output_dir=./saved_models \
 --model_type=roberta \
 --tokenizer_name=microsoft/codebert-base-mlm \
 --model_name_or_path=microsoft/codebert-base-mlm \
 --csv_store_path ./mhm_attack_otest_subs_2400_2800.csv \
 --base_model=microsoft/codebert-base-mlm \
 --train_data_file=../preprocess/dataset/train_subs.jsonl \
 --eval_data_file=../preprocess/dataset/test_subs_2400_2800.jsonl \
 --test_data_file=../preprocess/dataset/test_subs.jsonl \
 --block_size 512 \
 --original \
 --eval_batch_size 16 \
 --seed 123456 2>&1 | tee mhm_attack_o_subs_2400_2800.log &
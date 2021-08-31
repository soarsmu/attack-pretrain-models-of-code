CUDA_VISIBLE_DEVICES=7 python gi_attack.py     --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/graphcodebert-base     --model_name_or_path=microsoft/graphcodebert-base     --csv_store_path ./attack_no_gi_train_subs_0_2000.csv     --base_model=microsoft/graphcodebert-base     --eval_data_file=../preprocess/train_subs_0_2000.jsonl     --code_length 512     --data_flow_length 128     --eval_batch_size 8     --seed 123456  2>&1 | tee attack_no_gi_train_subs_0_2000.log &



CUDA_VISIBLE_DEVICES=0 python gi_attack.py     --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/graphcodebert-base     --model_name_or_path=microsoft/graphcodebert-base     --csv_store_path ./attack_no_gi_train_subs_2000_4000.csv     --base_model=microsoft/graphcodebert-base     --eval_data_file=../preprocess/train_subs_2000_4000.jsonl     --code_length 512     --data_flow_length 128     --eval_batch_size 8     --seed 123456  2>&1 | tee attack_no_gi_train_subs_2000_4000.log &



CUDA_VISIBLE_DEVICES=0 python gi_attack.py     --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/graphcodebert-base     --model_name_or_path=microsoft/graphcodebert-base     --csv_store_path ./attack_no_gi_train_subs_4000_6000.csv     --base_model=microsoft/graphcodebert-base     --eval_data_file=../preprocess/train_subs_4000_6000.jsonl     --code_length 512     --data_flow_length 128     --eval_batch_size 8     --seed 123456  2>&1 | tee attack_no_gi_train_subs_4000_6000.log &



CUDA_VISIBLE_DEVICES=1 python gi_attack.py     --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/graphcodebert-base     --model_name_or_path=microsoft/graphcodebert-base     --csv_store_path ./attack_no_gi_train_subs_6000_8000.csv     --base_model=microsoft/graphcodebert-base     --eval_data_file=../preprocess/train_subs_6000_8000.jsonl     --code_length 512     --data_flow_length 128     --eval_batch_size 8     --seed 123456  2>&1 | tee attack_no_gi_train_subs_6000_8000.log &



CUDA_VISIBLE_DEVICES=2 python gi_attack.py     --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/graphcodebert-base     --model_name_or_path=microsoft/graphcodebert-base     --csv_store_path ./attack_no_gi_train_subs_8000_10000.csv     --base_model=microsoft/graphcodebert-base     --eval_data_file=../preprocess/train_subs_8000_10000.jsonl     --code_length 512     --data_flow_length 128     --eval_batch_size 8     --seed 123456  2>&1 | tee attack_no_gi_train_subs_8000_10000.log &



CUDA_VISIBLE_DEVICES=2 python gi_attack.py     --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/graphcodebert-base     --model_name_or_path=microsoft/graphcodebert-base     --csv_store_path ./attack_no_gi_train_subs_10000_12000.csv     --base_model=microsoft/graphcodebert-base     --eval_data_file=../preprocess/train_subs_10000_12000.jsonl     --code_length 512     --data_flow_length 128     --eval_batch_size 8     --seed 123456  2>&1 | tee attack_no_gi_train_subs_10000_12000.log &



CUDA_VISIBLE_DEVICES=2 python gi_attack.py     --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/graphcodebert-base     --model_name_or_path=microsoft/graphcodebert-base     --csv_store_path ./attack_no_gi_train_subs_12000_14000.csv     --base_model=microsoft/graphcodebert-base     --eval_data_file=../preprocess/train_subs_12000_14000.jsonl     --code_length 512     --data_flow_length 128     --eval_batch_size 8     --seed 123456  2>&1 | tee attack_no_gi_train_subs_12000_14000.log &



CUDA_VISIBLE_DEVICES=0 python gi_attack.py     --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/graphcodebert-base     --model_name_or_path=microsoft/graphcodebert-base     --csv_store_path ./attack_no_gi_train_subs_14000_16000.csv     --base_model=microsoft/graphcodebert-base     --eval_data_file=../preprocess/train_subs_14000_16000.jsonl     --code_length 512     --data_flow_length 128     --eval_batch_size 8     --seed 123456  2>&1 | tee attack_no_gi_train_subs_14000_16000.log &



CUDA_VISIBLE_DEVICES=3 python gi_attack.py     --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/graphcodebert-base     --model_name_or_path=microsoft/graphcodebert-base     --csv_store_path ./attack_no_gi_train_subs_16000_18000.csv     --base_model=microsoft/graphcodebert-base     --eval_data_file=../preprocess/train_subs_16000_18000.jsonl     --code_length 512     --data_flow_length 128     --eval_batch_size 8     --seed 123456  2>&1 | tee attack_no_gi_train_subs_16000_18000.log &



CUDA_VISIBLE_DEVICES=3 python gi_attack.py     --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/graphcodebert-base     --model_name_or_path=microsoft/graphcodebert-base     --csv_store_path ./attack_no_gi_train_subs_18000_20000.csv     --base_model=microsoft/graphcodebert-base     --eval_data_file=../preprocess/train_subs_18000_20000.jsonl     --code_length 512     --data_flow_length 128     --eval_batch_size 8     --seed 123456  2>&1 | tee attack_no_gi_train_subs_18000_20000.log &



CUDA_VISIBLE_DEVICES=3 python gi_attack.py     --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/graphcodebert-base     --model_name_or_path=microsoft/graphcodebert-base     --csv_store_path ./attack_no_gi_train_subs_20000_22000.csv     --base_model=microsoft/graphcodebert-base     --eval_data_file=../preprocess/train_subs_20000_22000.jsonl     --code_length 512     --data_flow_length 128     --eval_batch_size 8     --seed 123456  2>&1 | tee attack_no_gi_train_subs_20000_22000.log &


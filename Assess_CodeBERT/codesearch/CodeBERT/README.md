# Scripts

```
python3 attack.py \
    --data_path '../data/CodeBERT/doc_code_dataset/' \
    --mlm_path 'microsoft/codebert-base' \
    --tgt_path './models/doc_code_codebert' \
    --output_dir data_defense/imdb_logs.tsv \
    --num_label 2 \
    --use_bpe 1 \
    --k 48 \
    --threshold_pred_score 0
```


```
python3 attack.py \
    --data_path '../data/CodeBERT/question_code_dataset/' \
    --mlm_path 'microsoft/codebert-base' \
    --tgt_path './models/question_code' \
    --output_dir data_defense/imdb_logs.tsv \
    --num_label 2 \
    --use_bpe 1 \
    --k 48 \
    --threshold_pred_score 0
```
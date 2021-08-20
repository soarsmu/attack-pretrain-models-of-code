
class Attacker():
    def __init__(self, args, codebert_tgt, tokenizer_tgt, codebert_mlm, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.codebert_tgt = codebert_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.codebert_mlm = codebert_mlm
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score



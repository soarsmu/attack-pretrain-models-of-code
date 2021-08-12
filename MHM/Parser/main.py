# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:57:54 2020

@author: DrLC
"""

import shutil
import os, sys
import pickle, gzip
import mytoken as tk
import build_dataset as bd
from pattern import StmtInsPos, DeclInsPos
from tqdm import tqdm


def dataset(dir='./tmp', tgt='./data/oj.pkl.gz',
            symtab='./data/oj_uid.pkl.gz',
            inspos_file='./data/oj_inspos.pkl.gz',
            done_file='dataset.done'):
    if tk.unzip():
        d = tk.tokenize()
        if d is not None:
            train, test = bd.split(d)
            idx2txt, txt2idx = bd.build_vocab(train['raw'])
            train_tokens = bd.text2index(train['raw'], txt2idx)
            test_tokens = bd.text2index(test['raw'], txt2idx)
            uids = []
            for _uids in train["uids"]:
                for _uid in _uids.keys():
                    if _uid not in uids:
                        uids.append(_uid)
            if not os.path.isfile(os.path.join(dir, done_file)):
                data = {"raw_tr": train["raw"], "y_tr": train["labels"],
                        "x_tr": train_tokens,
                        "raw_te": test["raw"], "y_te": test["labels"],
                        "x_te": test_tokens,
                        "idx2txt": idx2txt, "txt2idx": txt2idx}
                uid = {"tr": train["uids"], "te": test["uids"], "all": uids}
                with gzip.open(tgt, "wb") as f:
                    pickle.dump(data, f)
                with gzip.open(symtab, "wb") as f:
                    pickle.dump(uid, f)
                with open(os.path.join(dir, done_file), "wb") as f:
                    pass
                stmt_poses_tr = [StmtInsPos(tr) for tr in tqdm(train['raw'])]
                stmt_poses_te = [StmtInsPos(te) for te in tqdm(test['raw'])]
                decl_poses_tr = [DeclInsPos(tr) for tr in tqdm(train['raw'])]
                decl_poses_te = [DeclInsPos(te) for te in tqdm(test['raw'])]
                inspos = {"stmt_tr": stmt_poses_tr, "stmt_te": stmt_poses_te,
                          "decl_tr": decl_poses_tr, "decl_te": decl_poses_te}
                with gzip.open(inspos_file, "wb") as f:
                    pickle.dump(inspos, f)
                    # shutil.rmtree(dir)


if __name__ == "__main__":
    dataset()

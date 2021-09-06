# Attack pre-trained models of code

This is the codebase for attacking pre-trained models of code.

All datasets, models and results can be downloaded from https://drive.google.com/uc?id=1mWSVewDUa_L_KdEczhyleM0XOD-hji9K.

For the source code files:

```
CodeXGLUE ----------------------------------- for attacking CodeBERT
    -- Authorship-Attribution
        -- attack.py ----------------------------- main function to attack CodeBERT
        -- attacker.py --------------------------- util functions to attack CodeBERT
        -- mhm.py -------------------------------- main function to attack CodeBERT by MHM
        -- model.py ------------------------------ CodeBERT downstream model
        -- run.py -------------------------------- finetune CodeBERT
        -- get_res.py ---------------------------- process .csv data to get result
    -- Clone-detection-BigCloneBench
        -- Same as the above
    -- Defect-detection
        -- Same as the above
GraphCodeBERT ----------------------------------- for attacking GraphCodeBERT
    -- Authorship-Attribution
        -- Same as the above
    -- Clone-detection-BigCloneBench
        -- Same as the above
    -- Defect-detection
        -- Same as the above
python_parser ------------------------ parser to extract data flows and identifiers
UserStudy ---------------------------- functions to process user study data
```

The commands for running each code scripts can be found in `README.md` under each subfolder. Please use a docker container with `pytorch/pytorch:1.5-cuda10.1-cudnn7-devel`.

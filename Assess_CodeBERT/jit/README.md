# JIT defect prediction

## Dependency

Python >=3.6.9

pip install torch

pip install transformers

## data & pre-trained models

data and pre-trained models can be found here (https://drive.google.com/drive/folders/199qFIMUg1rm53jnKejCDKxqMFY89wzNd)

## Training

    $$ python main.py -train -train_data [path of training data] -save-dir [where to save the trained model] -dictionary_data [path of the dict]

Example:

    $$ python main.py  -train -train_data './data/op_data/openstack_train_changed.pkl'  -save-dir './trained_model_op'  -dictionary_data './data/op_dict.pkl'
        

## Validation
    $$ python main.py -train -train_data [path of training data] -save-dir [where to save the trained model] -dictionary_data [path of the dict] -valid -load_model [snapshot of model needed to do validation]    

Example:

    $$  python main.py  -train -train_data './data/op_data/openstack_train_changed.pkl'  -save-dir './trained_model_op'  -dictionary_data './data/op_dict.pkl' -valid -load_model 'trained_model_op/2021-07-30_04-42-06/epoch_1_step_600.pt' 


## Testing

    $$ python main.py -predict -pred_data [path of testing data]  -load_model [path of trained model] -dictionary_data [path of the dict]
    
Example:

    $$ python main.py  -predict -pred_data './qt_data/qt_test_changed.pkl'  -load_model './trained_model_qt/epoch_1_step_750.pt'  -dictionary_data './qt_dict.pkl'

# Few-Shot Incremental Learning for NeurIPS2022

## Which files to focus
- When you add some arguments

        train.py

- When you modify your model

        model/cec/Network.py

- When you modify training or test processes

        model/cec/fscil_trainer.py

- When you modify the feature extractor

        model/resnet20_cifar100.py

- etc. -> not recommended to modify

- Ignore some python files xx_tmp.py

## Requirements
You should already have some dirs or files such as ./params, ./data, etc.

Please refer to the original CEC repo for these requirements (https://github.com/icoz69/CEC-CVPR2021)

pre-trained parameters of different model architectures can be downloaded in (https://drive.google.com/drive/folders/1I0orLc0RWtGbZRtQGWo72LgCUcoZgFd0?usp=sharing)

Put them in ./params/cifar100 and use it by modifying -model_dir argument and resnet20_cifar100.py.

## Usage
Refer to the below script when you run your codes.

Make script.py and execute it by python script.py

You can slightly modify the script for additional hyperparameters or differen pre-trained model ()

--save aregument is for the name of the experiment.

When you run some codes with different version, modify the first part of the name (i.e. newmodelencoder -> your version name).


    import os
    gpu = 0

    query_new_lamb = 0.03
    inner_steps = 1
    outer_lr = 0.02
    proto_lamb = 0
    contrastive = 0
    fc_lamb = 0
    num_query_new = 20
    num_query_base = 10
    num_query_base_class = 10

    os.system(f'python train.py -project cec -dataset cifar100 -epochs_base 100 -episode_way 5 -episode_shot 5 -low_way 5 -low_shot 5 -step 20 -gamma 0.5 -lr_base 0.002 -gpu {gpu} -model_dir params/cifar100/session0_max_acc.pth --meta --save_fig false --proto_lamb {proto_lamb} --lamb_contrastive {contrastive} --new_loader --dropcontrastive_lamb 0 --mixup --outer_lr {outer_lr} --fullproto --inner_steps {inner_steps} --freeze_epoch {freeze_epoch} --query_new_lamb {query_new_lamb} --save newmodelencoder_innersteps{inner_steps}_outerlr{outer_lr}_fullproto{proto_lamb}_contrastive{contrastive}_fc{fc_lamb}_querynew{query_new_lamb}_second_qbc{num_query_base_class}_qb{num_query_base}_qn{num_query_new} --num_query_base_class {num_query_base_class} --num_query_base {num_query_base} --num_query_new {num_query_new} --second_order_maml --fc_lamb {fc_lamb}')
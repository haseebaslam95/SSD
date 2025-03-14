## Learning from Stochastic Teacher Representations Using Student-Guided Knowledge Distillationng 



## Abstract


<!-- <div style="text-align:center">
  <img src=/>
</div> -->

Advances in self-distillation have shown that when knowl-
edge is distilled from a teacher network to a student network with the
same architecture, the student performance surpasses the teacher be-
cause it leads to a richer feature representation. Alternatively, ensembling
multiple models also improves performance. However, training, deploy-
ing, and storing multiple models for ensembling becomes impractical as
the size increases. Even distilling an ensemble to a single model first re-
quires separate training of multiple teacher models and does not fully
leverage the inherent stochasticity in neural networks. These constraints
are particularly prohibitive in resource-constrained or latency-sensitive
applications such as wearable time series. This paper proposes to train
only one model and generate multiple diverse teacher representations us-
ing distillation-time dropout. However, generating these representations
stochastically leads to a grave problem of noisy/misinforming representa-
tions. To overcome this problem, a novel stochastic self-distillation (SSD)
training strategy is introduced for filtering and weighting teacher repre-
sentation to distill from task-relevant representations only, using student-
guided knowledge distillation (SGKD). The student representation at
each distillation step is used as authority to guide the distillation pro-
cess. Experimental results on real-world affective computing (StressID
and Biovid Pain), wearable/biosignal datasets from the UCR Archive,
HAR dataset, and image classification datasets show that the proposed
SSD method can outperform state-of-the-art methods without increas-
ing the model size at deployment and at train time, incurs negligible
additional computational cost compared to existing approaches.


## Directory Structure
```python
SSD
|   environment.yml # Requirements for conda environment
|   LICENSE
|   README.md
|   requirments.txt # Requirements for pip environment
|   
|
|   batch_distillation.sh # Training script if you wish to run from terminal
|   main_distillation.py # Training script if you with to run all folds from .py file       
|       
\---src

    |   config.json # Training configurations
    |   logger_config.json # Logger configurations
    |   ssd_utils.py # Provides all the functionalites used in SGKD
    |   parser.py # Parser for training configurations
    |   Tdistillation.py # Main training script
    |  __init__.py
    |   
    +---models
    |   |   main_painAttnNet.py # Main model wrapper
    |   |   module_mscn.py* # Multiscale convolutional network
    |   |   module_se_resnet.py # Squeeze-and-excitation residual network
    |   |   module_transformer_encoder.py # Transformer encoder block
    |   \   __init__.py
    |           
    +---tests # Unit tests
    |   |   test_generate_kfolds_index.py
    |   |   test_mscn.py
    |   |   test_PainAttnNet_output.py
    |   |   test_process_bioVid.py
    |   |   test_se_resnet.py
    |   |   test_transformer_encoder.py
    |   \   __init__.py
    |           
    +---trainers # Training modules
    |   |   checkpoint_handler.py # Checkpoint handler
    |   |   device_prep.py # Device preparation, CPU or GPU
    |   |   main_trainer.py # Main trainer scripts
    |   |   metrics_manager.py # Metrics manager and other metrics functions
    |   \   __init__.py
    |           
    \---utils
        |   process_bioVid.py # Data processing for BioVid
        |   utils.py # Other utility functions
        |   extractdistillresults.py # Provides functionality to read all the log files to calcluate average and per fold plots
        \   __init__.py

```  



## Teacher Training

Follow all the instructions provided in the Original PainAttentionNet Repository to setup and preprocess Biovid Dataset. 
The per fold models trained should be stored to be later used at teacher models in student training.

Link to PAN Repository: https://github.com/zhenyuanlu/PainAttnNet/tree/main


### Training Student Model with Script
```
sh batch_distillation.sh
```
### Training Student Model from Python file
```
python main_distillation.py 
```

You can change settings in `config.py` for training configurations. Also update required paths in all files.


## Dataset
The dataset is available at [BioVid Heat Pain Database](https://www.nit.ovgu.de/BioVid.html).

## Reference


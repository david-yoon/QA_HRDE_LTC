

## This repository contains the source code & data corpus for the models used in the following paper,

**Learning to Rank Question-Answer Pairs using Hierarchical Recurrent Encoder with Latent Topic Clustering**, NAACL-18, <a href="http://aclweb.org/anthology/N18-1142">paper</a>


----------


### [download dataset]

- data corpus is available from "releases" tab
- place each data corpus into following path of the project
 
>     / data / ubuntu_v1 /
>            / ubuntu_v2 /
>            / samsungQA /

- Note that ubuntu_v1/v2 are originally from following github repository.
<a href="https://github.com/npow/ubottu">ubuntu-v1</a>, 
<a href="https://github.com/rkadlec/ubuntu-ranking-dataset-creator">ubuntu-v2</a>
----------
### [source code path]

>     / data           : contains dataset (ubuntu v1/v2, samsungQA)
>     / src_ubuntu_v1  : source code for ubuntu v1 data
>     / src_ubuntu_v2  : source code for ubuntu v2 data
>     / src_samsungQA  : source code for samsung QA data

----------
### [Training]
- each source code folder contains training script
  << for example >>
>     /src_ubunutu_v1/
>     ./run_RDE.sh      : train ubuntu_v1 dataset with RDE model
>     ./run_RDE_LTC.sh  : train ubuntu_v1 dataset with RDE-LTC model
>     ./run_HRDE.sh     : train ubuntu_v1 dataset with HRDE model
>     ./run_HRDE_LTC.sh : train ubuntu_v1 dataset with HRDE-LTC model
- best model will be stored in save folder
  << for example >>
>     /src_ubunutu_v1/save/
   


----------


### [Inference]

- each source code folder contains inference code   
   << execution example >>
   /src_ubunutu_v1/
>      python eval_RDE.py       : inference ubuntu_v1 testset with RDE model
>      python eval_RDE_LTC.py   : inference ubuntu_v1 testset with RDE-LTC model
>      python eval_HRDE.py      : inference ubuntu_v1 testset with HRDE model
>      python eval_HRDE_LTC.py  : inference ubuntu_v1 testset with HRDE-LTC model
      
- inference code use saved model in 'save' folder 
- inference result will be stored in 'save' folder
   << example >>
>     /src_ubunutu_v1/save/result_RDE.txt


----------


### [cite]

- Please cite our paper, when you use our code | dataset | model.

>     @inproceedings{yoon2018learning, 
>        title={Learning to Rank Question-Answer Pairs Using Hierarchical Recurrent Encoder with Latent Topic Clustering}, 
>        author={Yoon, Seunghyun and Shin, Joongbo and Jung, Kyomin}, 
>        booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)}, 
>        volume={1},
>        pages={1575--1584},
>        year={2018} 
>        }   

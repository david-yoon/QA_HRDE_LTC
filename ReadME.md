
[download dataset]
- please download data file from following url and store data in following path
- data file size is 2GB
  https://drive.google.com/drive/folders/1nGoe0MHiZ636GixOWVgQuXJG7YIk5ag5?usp=sharing


naacl_code / data / ubuntu_v1 /
		  / ubuntu_v2 /
 		  / samsungQA /



[source code path]

naacl_code / data           : contains dataset (ubuntu v1/v2, samsungQA)
           / src_ubuntu_v1  : source code for ubuntu v1 data
           / src_ubuntu_v2  : source code for ubuntu v2 data
           / src_samsungQA  : source code for samsung QA data


[training]
- each source code folder contains training script
  << for example >>
   naacl_code/src_ubunutu_v1/
      run_RDE.sh      : train ubuntu_v1 dataset with RDE model
      run_RDE_LTC.sh  : train ubuntu_v1 dataset with RDE-LTC model
      run_HRDE.sh     : train ubuntu_v1 dataset with HRDE model
      run_HRDE_LTC.sh : train ubuntu_v1 dataset with HRDE-LTC model
- best model will be stored in save folder
  << for example >>
   naacl_code/src_ubunutu_v1/save/
   

[inference]
- each source code folder contains inference code   
   << execution example >>
   naacl_code/src_ubunutu_v1/
      python eval_RDE.py          : inference ubuntu_v1 testset with RDE model
      python eval_RDE_LTC.py   : inference ubuntu_v1 testset with RDE-LTC model
      python eval_HRDE.py        : inference ubuntu_v1 testset with HRDE model
      python eval_HRDE_LTC.py : inference ubuntu_v1 testset with HRDE-LTC model
      
- inference code use saved model in 'save' folder 
- inference result will be stored in 'save' folder
   << example >>
   naacl_code/src_ubunutu_v1/save/result_RDE.txt

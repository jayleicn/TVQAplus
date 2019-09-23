## TVQA+: Spatio-Temporal Grounding for Video Question Answering

![qa_example](imgs/qa_example_pair.png)

We present the task of Spatio-Temporal Video Question Answering, which requires intelligent 
systems to simultaneously retrieve relevant moments and detect referenced visual concepts 
(people and objects) to answer natural language questions about videos. 
We first augment the [TVQA](http://tvqa.cs.unc.edu/) dataset with 310.8k bounding boxes, linking depicted objects to 
visual concepts in questions and answers. 
We name this augmented version as TVQA+.
We then propose Spatio-Temporal Answerer with Grounded Evidence (STAGE), 
a unified framework that grounds evidence in both the spatial and temporal domains to 
answer questions about videos. 
Comprehensive experiments and analyses demonstrate the effectiveness of our framework and 
how the rich annotations in our TVQA+ dataset can contribute to the question answering task. 
As a side product, by performing this joint task, our model is able to produce more insightful 
intermediate results. 


In this repository, we provide PyTorch Implementation of the STAGE model, along with basic 
preprocessing and evaluation code for TVQA+ dataset.


[TVQA+: Spatio-Temporal Grounding for Video Question Answering](https://arxiv.org/abs/1904.11574)<br>
[Jie Lei](http://www.cs.unc.edu/~jielei/),  [Licheng Yu](http://www.cs.unc.edu/~licheng/), 
[Tamara L. Berg](http://Tamaraberg.com), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/). 
   [[PDF]](https://arxiv.org/abs/1904.11574) [[TVQA+ Dataset]](http://tvqa.cs.unc.edu)


### Model
- **STAGE Overview**. Spatio-Temporal Answerer with Grounded Evidence (STAGE), a unified framework that grounds evidence in both the spatial and temporal domains to answer questions about videos.  
![model_overview](imgs/model_overview.png)


- **Prediction Examples**
![example_predictions](imgs/model_prediction.png) 


### Requirements
- Python 2.7
- PyTorch 1.1.0 (should work for 0.4.0 - 1.2.0)
- tensorboardX
- tqdm
- h5py
- numpy


### Training and Evaluation
1, Download preprocessed features from [Google Drive](https://drive.google.com/drive/folders/1eTy69AgdJNs-bL_fNLlcC5pMS_QKowrf?usp=sharing). 
We recommend using [gdrive](https://github.com/prasmussen/gdrive) to download it from command line. When finished, move it to the root of this project, make sure `release_path` in `run_main.sh` is pointing to the feature directory.

2, Run in `debug` mode to test your environment, path settings:
```
bash run_main.sh debug
```

3, Train the full STAGE model:
```
bash run_main.sh --add_local
```
note you will need around 50 GB of memory to load the data. Otherwise, you can additionally add `--no_core_driver` flag to stop loading 
all the features into memory. After the training, you should be able to get ~72.00% QA Acc, which is comparable to the reported number. 

4, Evaluation (TODO)


### Citation
```
@inproceedings{lei2019tvqa,
  title={TVQA+: Spatio-Temporal Grounding for Video Question Answering},
  author={Lei, Jie and Yu, Licheng and Berg, Tamara L and Bansal, Mohit},
  booktitle={Tech Report, arXiv},
  year={2019}
}
```

### TODO
1. [x] Add data preprocessing scripts (provided preprocessed features)
2. [x] Add model and training scripts
3. [ ] Add inference and evaluation scripts


### Contact
- Dataset: faq-tvqa-unc [at] googlegroups.com
- Model: Jie Lei, jielei [at] cs.unc.edu

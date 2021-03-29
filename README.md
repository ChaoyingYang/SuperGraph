## SuperGraph
Code release for paper entitled "SuperGraph: Spatial-temporal graph-based feature extraction for rotating machinery diagnosis", and this paper is revised in IEEE Transactions on Industrial Electronics.

## Guide
This project only provides the data and related codes used in the paper that allows anyone to reproduce all results presented in the paper. All algorithms were written by python 3.8, PyTorch 1.7.0, and PyTorch geometric  through running on a computer with an Intel Core 298 i7-8700K CPU and a 32G RAM.

## Requirements
- Python 3.8  
- Numpy 1.16.2  
- sklearn 0.21.3  
- Scipy 1.2.1   
- [Pytorch 1.7.0 ](https://pytorch.org/)
- [Pytroch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)


## Datasets
- **[CWRU Bearing Dataset from Case Western Reserve University](https://csegroups.case.edu/bearingdatacenter/pages/download-data-file/)**
- **[PU Bearing Dataset (KAT dataset)](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/)**
- Gearbox dataset

## Pakages

This repository is organized as:
- [Code](https://github.com/ChaoyingYang/SuperGraph/tree/master/code) contains the corresonding codes.
- [data](https://github.com/ChaoyingYang/SuperGraph/tree/master/data) contains the used data from CWRU dataset， KAT dataset, and Gearbox dataset.


## Usage
- Download the datasets  
- Modify the file storage address in the code  
- Run the corresponding codes  
  


## Citation
Codes:
```
@misc{Yang2021,  
author = {Chaoying Yang, Kaibo Zhou, and Jie Liu},  
title = {SuperGraph: Spatial-temporal graph-based feature extraction for rotating machinery diagnosis},  
year = {2021},  
publisher = {GitHub},  
journal = {GitHub repository},  
howpublished = {\url{https://github.com/ChaoyingYang/SuperGraph}},  
}  
```
Paper:
```
@article{Yang2021,
  title={SuperGraph: Spatial-temporal graph-based feature extraction for rotating machinery diagnosis},
  author={Chaoying Yang, Kaibo Zhou, and Jie Liu},
  journal={},
  year={2021}
}
```

## Contact
Chaoying Yang - yangcy@hust.edu.cn 
 
Jie Liu - jie_liu@hust.edu.cn


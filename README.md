# NARRE

This is our implementation for the paper:



*Chong Chen, Min Zhang, Yiqun Liu, and Shaoping Ma. 2018. [Neural Attentional Rating Regression with Review-level Explanations.](http://www.thuir.cn/group/~YQLiu/publications/WWW2018_CC.pdf) 
In WWW'18.*

**Please cite our WWW'18 paper if you use our codes. Thanks!**

```
@inproceedings{chen2018neural,
  title={Neural Attentional Rating Regression with Review-level Explanations},
  author={Chen, Chong and Zhang, Min and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of the 2018 World Wide Web Conference on World Wide Web},
  pages={1583--1592},
  year={2018},
}
```

Author: Chong Chen (cstchenc@163.com)

## Environments

- python 2.7
- Tensorflow (version: 0.12.1)
- numpy
- pandas


## Dataset

In our experiments, we use the datasets from  Amazon 5-core(http://jmcauley.ucsd.edu/data/amazon) and Yelp Challenge 2017(https://www.yelp.com/dataset_challenge).

## Example to run the codes		

Data preprocessing:

```
python loaddata.py	
python data_pro.py
```

Train and evaluate the model:

```
python train.py
```


Last Update Date: April 14, 2018

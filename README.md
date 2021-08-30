## [VisualNews : Benchmark and Challenges in Entity-aware Image Captioning](https://arxiv.org/abs/2010.03743)
Fuxiao Liu, [Yinghan Wang](https://www.linkedin.com/in/yinghan-wang-39980a119/), [Tianlu Wang](http://www.cs.virginia.edu/~tw8cb/), [Vicente Ordonez](https://www.vicenteordonez.com/)

### Abstract 
In this paper we propose VisualNews-Captioner, an entity-aware model for the task of news image captioning. We also introduce VisualNews, a large-scale benchmark consisting of more than one million news images along with associated news articles, image captions, author information, and other metadata. Unlike the standard image captioning task, news images depict situations where people, locations, and events are of paramount importance. Our proposed method is able to effectively combine visual and textual features to generate captions with richer information such as events and entities. More specifically, we propose an Entity-Aware module along with an Entity-Guide attention layer to encourage more accurate predictions for named entities. Our method achieves state-of-the-art results on both the GoodNews and VisualNews datasets while having significantly fewer parameters than competing methods. Our larger and more diverse VisualNews dataset further highlights the remaining challenges in captioning news images.

### Getting Data
- Our dataset is available upon request. 
- To access our dataset, please refer to this [demo](./VisualNews-Dataset.ipynb)
![Examples from our VisualNews dataset](./sample.jpg)

### Diversity Analysis

 - Statistics of named entities of four news agencies in Visual News dataset. For example, The Guardian agency has 602,572 captions and 435,629 PERSON entities, thus each caption will on average have 0.72 PERSON entities. However, each caption only has 0.46 PERSON entities on average in BBC agency. This difference also demonstrates the diversity of our dataset.

   |                                     |       Guardian |             BBC |        USA TODAY|   WashingtonPost|
   | ----------------------------------- | :-------------:| :--------------:| :--------------:| :--------------:|
   | Number of Captions                  |         602572 |          198186 |          151090 |          128744 |
   | Number of PERSON_                   |    435629/0.72 |      92758/0.46 |     127548/0.84 |      89811/0.69 | 
   | Number of ORG_                      |    225296/0.37 |      56783/0.29 |      86276/0.57 |      55489/0.43 | 
   | Number of GPE_                      |    224039/0.37 |      56897/0.29 |      78653/0.52 |      63790/0.50 | 
   | Number of DATE_                     |    140325/0.23 |      43938/0.22 |      71266/0.47 |      37528/0.29 |  
   | Number of CARDINAL_                 |     77068/0.13 |      27729/0.14 |      25992/0.17 |      11719/0.09 |   
   | Number of FAC_                      |     38807/0.06 |       7628/0.04 |      23847/0.15 |      14508/0.11 |   
   | Number of NORP_                     |     62865/0.10 |      19997/0.10 |      16103/0.11 |      20124/0.16 | 
   | Number of ORDINAL_                  |     22140/0.04 |       5268/0.03 |      13177/0.09 |       4927/0.04 | 
   | Number of LOC_                      |     14215/0.02 |       2181/0.01 |       9664/0.06 |       2928/0.02 |
   | Number of PRODUCT_                  |     24268/0.04 |       6549/0.03 |       8907/0.06 |       6680/0.05 | 
   | Number of TIME_                     |     12116/0.02 |       3259/0.02 |       5220/0.03 |       3134/0.02 | 
   | Number of WORK_OF_ART_              |      8664/0.01 |       1500/0.01 |       2953/0.02 |       1810/0.01 | 
   | Number of QUANTITY_                 |      4077/0.01 |       1658/0.01 |       1414/0.01 |        957/0.01 | 
   | Number of LAW_                      |      1977/0.00 |        347/0.00 |       1083/0.01 |        698/0.01 | 
   | Number of MONEY_                    |      1022/0.00 |        350/0.00 |        144/0.00 |         93/0.00 | 
   | Number of PERCENT_                  |        81/0.00 |         17/0.00 |         64/0.00 |        184/0.00 | 
   | Number of LANGUAGE_                 |       920/0.00 |        323/0.00 |         57/0.00 |        127/0.00 |

 - CIDEr scores of the same basic captioning model on different train (row) and test (columns) splits. News images and captions from different agencies have different characters, leading to a performance decrease when training set and test set are not from the same agency.

   |                                     |       Guardian |             BBC |        USA TODAY|   WashingtonPost|
   | ----------------------------------- | :-------------:| :--------------:| :--------------:| :--------------:|
   | Guardian                            |            1.0 |             0.6 |             0.6 |             0.7 |
   | BBC                                 |            1.9 |             1.6 |             1.7 |             0.7 | 
   | USA TODAY                           |            1.3 |             1.2 |             3.7 |             2.7 |
   | WashingtonPost                      |            1.2 |             1.2 |             2.0 |             2.5 | 
   
   
 - Select 50000 captions from each source. USA TODAY has 17013 unique 'PERSON' entities, WashingtonPost has 16261, BBC has 12726 and Guardian has 17745. Statistics of named entities from each source and calculate their overlap. Statistics of other named entities will be updated soon.

   | 'PERSON'                            |       Guardian |             BBC |        USA TODAY|   WashingtonPost|
   | ----------------------------------- | :-------------:| :--------------:| :--------------:| :--------------:|
   | Guardian                            |          17745 |        **2345** |            2048 |            1997 |
   | BBC                                 |       **2345** |           12726 |            1297 |            1413 | 
   | USA TODAY                           |           2048 |            1297 |           17013 |        **2957** |
   | WashingtonPost                      |           1997 |            1413 |        **2957** |           16261 | 
   

### Requirements
- Python 3
- Pytorch > 1.0

This repo is under construction and we will add the code shortly. If you have any questions, please email: fl3es@virginia.edu

### Citing
If you find our paper/code useful, please consider citing:

```
@misc{liu2020visualnews,
      title={VisualNews : Benchmark and Challenges in Entity-aware Image Captioning}, 
      author={Fuxiao Liu and Yinghan Wang and Tianlu Wang and Vicente Ordonez},
      year={2020},
      eprint={2010.03743},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

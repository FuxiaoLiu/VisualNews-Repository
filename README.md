## [VisualNews : Benchmark and Challenges in Entity-aware Image Captioning](https://arxiv.org/abs/2010.03743)
Fuxiao Liu, [Yinghan Wang](https://www.linkedin.com/in/yinghan-wang-39980a119/), [Tianlu Wang](http://www.cs.virginia.edu/~tw8cb/), [Vicente Ordonez](https://www.vicenteordonez.com/)

### Abstract 
In this paper we propose VisualNews-Captioner, an entity-aware model for the task of news image captioning. We also introduce VisualNews, a large-scale benchmark consisting of more than one million news images along with associated news articles, image captions, author information, and other metadata. Unlike the standard image captioning task, news images depict situations where people, locations, and events are of paramount importance. Our proposed method is able to effectively combine visual and textual features to generate captions with richer information such as events and entities. More specifically, we propose an Entity-Aware module along with an Entity-Guide attention layer to encourage more accurate predictions for named entities. Our method achieves state-of-the-art results on both the GoodNews and VisualNews datasets while having significantly fewer parameters than competing methods. Our larger and more diverse VisualNews dataset further highlights the remaining challenges in captioning news images.

### Getting Data
- Our dataset is available upon request. 
- To access our dataset, please refer to this [demo](./VisualNews-Dataset.ipynb)
- ![Examples from our VisualNews dataset](./sample.jpg)

### Dataset Analysis

|                                     |       Guardian |             BBC |        USA TODAY|.  WashingtonPost|
| ----------------------------------- | :-------------:| :--------------:| :--------------:| :--------------:|
| Number of Caption                   |         602572 |          198186 |          151090 |          128744 |
| Number of PERSON_                   |    435629/0.72 |      92758/0.46 |     127548/0.84 |      89811/0.69 | 
| Number of ORG_                      |           LSTM |                 |                 |                 | 
| Number of GPE_                      |           LSTM |                 |                 |                 | 
| Number of DATE_                     |           LSTM |                 |                 |                 | 
| Number of CARDINAL_                 |           LSTM |                 |                 |                 | 
| Number of FAC_                      |           LSTM |                 |                 |                 | 
| Number of NORP_                     |           LSTM |                 |                 |                 | 
| Number of ORDINAL_                  |           LSTM |                 |                 |                 | 
| Number of LOC_                      |           LSTM |                 |                 |                 | 
| Number of PRODUCT_                  |           LSTM |                 |                 |                 | 
| Number of TIME_                     |           LSTM |                 |                 |                 | 
| Number of WORK_OF_ART_              |           LSTM |                 |                 |                 | 
| Number of QUANTITY_                 |           LSTM |                 |                 |                 | 
| Number of LAW_                      |           LSTM |                 |                 |                 | 
| Number of MONEY_                    |           LSTM |                 |                 |                 | 
| Number of PERCENT_                  |           LSTM |                 |                 |                 | 
| Number of LANGUAGE_                 |           LSTM |                 |                 |                 | 




### Requirements
- Python 3
- Pytorch > 1.0

### Training
```sh
# Train the full model on VisualNews.
CUDA_VISIBLE_DEVICES=0 python main.py
```
I will update the code ASAP. If you have any questions, please email: fl3es@virginia.edu

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

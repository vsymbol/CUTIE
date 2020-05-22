# CUTIE

TensorFlow implementation of the paper "CUTIE: Learning to Understand Documents with Convolutional Universal Text Information Extractor."
Xiaohui Zhao [ArXiv 2019](https://arxiv.org/abs/1903.12363v4)

## Results

Result evaluated on 4,484 receipt documents, including taxi receipts, meals entertainment receipts, and hotel receipts, with 9 different key information classes. (AP / softAP)
|Method     | #Params   |  Taxi         |  Hotel        |
| ----------|:---------:| :-----:       | :-----:       |
| CloudScan | -         |  82.0 / -     |  60.0 / -     |
| BERT      | 110M      |  88.1 / -     |  71.7 / -     |
| CUTIE     |**14M**    |**94.0 / 97.3**|**74.6 / 87.0**|

![Taxi](https://github.com/vsymbol/CUTIE/others/example_1.jpg)

![Hotel](https://github.com/vsymbol/CUTIE/others/example_2.jpg)


## Installation & Usage

```
pip install -r requirements.txt
```

1. Generate your own dictionary with main_build_dict.py / main_data_tokenizer.py
2. Train your model with main_train_json.py

CUTIE achieves best performance with rows/cols well configured. For more insights, refer to statistics in the file (others/TrainingStatistic.xlsx).

![Chart](https://github.com/vsymbol/CUTIE/others/chart.jpg)


## TLDR

For information about the input example, refer to [issue discussion](https://github.com/vsymbol/CUTIE/issues/7).

The project is refreshed with all history removed. All programs are runnable expect that the data example is not uploaded.
Since the project was built in my previous workplace, the data format can not be uploaded without permission right now. However, you may infer the correct data format from the data_loader_json.py file. Pull request is welcomed for making the project runnable out of the box.

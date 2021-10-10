# Punctuation Restoration Exercise

The transcripts generated through ASR are often not well punctuated. In this exercise, we are dealing with the problem of placing punctuations at proper place in unpunctuated text. I have trained this model for these 4 punctuations:

1. APOSTROPHE
2. PERIOD
3. QUESTION_MARK
4. COMMA


## Dataset

We are using open-source available dataset from [here](https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2). It has over 1.5 million punctuated sentences.


## Architecture

We are formulating this problem as token-level classification problem. We are using pre-trained model BERT here, but you can try other models as well by adding them to [config](src/config.py).

We have put a linear layer on top of BERT output. Having information about what punctuations have been seen so far will help in predicting current output, hence we are trying with a CRF layer too.

More layers/models can be simply added in [here](src/model.py)

## How to Train
1. Create a virtualenv and download [requirements](requirements.txt) 
2. Make changes to [config](src/config.py) as per which punctuations you are training for and which pre-trained model you want to use.
3. Training  supports a number of arguments which you can find [here](src/train.py).
	A sample command: `python src/train.py --epoch=3 --save-path=output_with_crf --use-crf=True`

## How to inference
1. Keep your test data in a file. You can use [this](data/test.txt). Format is one test case per line.
2. Run inference using below command.
`python src/inference.py --weight-path=output/weights.pt ----in-file=data/test.txt` 

## Our results

**Without CRF**

| Predictions (left) | O | APOSTROPHE | PERIOD | Q. MARK | COMMA |
| ------------- | ------------- |  ------------- |  ------------- |  ------------- |------------- |
| **Actual (Down)**  |  |  |  |  |  |
| Precision  | 99.74 | 99.86 | 99.21 | 98.33 | 76.93
| Recall  | 99.68 | 99.68 | 99.21 | 98.11 | 81.03
| F1  | 99.71 | 99.77 | 99.21 | 98.22 | 78.93 |

**With CRF**

| Predictions (left) | O | APOSTROPHE | PERIOD | Q. MARK | COMMA |
| ------------- | ------------- |  ------------- |  ------------- |  ------------- |------------- |
| **Actual (Down)**  |  |  |  |  |  |
| Precision  | 99.74 | 99.86 | 99.21 | 98.33 | 76.93
| Recall  | 99.68 | 99.68 | 99.21 | 98.11 | 81.03
| F1  | 99.71 | 99.77 | 99.21 | 98.22 | 78.93 |

# NER_Chinese_medical
Chinese medical named entity recognition using roberta + Flat model + adversarial training.  
Particiate in [CBLUE(Chinese Biomedical Language Understanding Evaluation)](https://tianchi.aliyun.com/cblue).
# Code description
- src/models.py -- base models including BERT+CRF, BERT+Linear, BERT+nested_Linear
- src/run_cmeee.py  -- main function for running the training process including output the prediction on the test set
- src/metrics.py  -- F1 metric computation for non-nested/ nested nework
- src/flat.py  -- FLAT model to include information from word
Other files are tools.


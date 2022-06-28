
import json
import logging
import pickle
import re
from collections import Counter
from itertools import repeat
from os.path import join, exists
from typing import List
from transformers import BertTokenizer, BertModel
import torch
import pkuseg
import numpy as np
import torch
from torch.utils.data import Dataset
import gensim
from gensim.models import KeyedVectors,word2vec
logger = logging.getLogger(__name__)
pku_seg = pkuseg.pkuseg(model_name='medicine')
NER_PAD, NO_ENT = '[PAD]', 'O'

LABEL1 = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'dis', 'bod']  # 按照出现频率从低到高排序
LABEL2 = ['sym']

LABEL  = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'] 

# 标签出现频率映射，从低到高
_LABEL_RANK = {L: i for i, L in enumerate(['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'])}

EE_id2label1 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL1 for P in ("B", "I")]
EE_id2label2 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL2 for P in ("B", "I")]
EE_id2label  = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL  for P in ("B", "I")]
print(EE_id2label1)
print(EE_id2label2)
EE_label2id1 = {b: a for a, b in enumerate(EE_id2label1)}

EE_label2id2 = {b: a for a, b in enumerate(EE_id2label2)}

EE_label2id  = {b: a for a, b in enumerate(EE_id2label)}

EE_NUM_LABELS1 = len(EE_id2label1)
EE_NUM_LABELS2 = len(EE_id2label2)
EE_NUM_LABELS  = len(EE_id2label)

def get_lattice_word(text):
    lattice_word = pkuseg_cut(text)
    # 按 词长度 升序   按 结束位置 升序
    lattice_word.sort(key=lambda x: len(x[0]))
    lattice_word.sort(key=lambda x: x[2])
    return lattice_word

def is_all_chinese(word_str):
    for c in word_str:
        if not '\u4e00' <= c <= '\u9fa5':
            return False
    return True

def pkuseg_cut(text):
    "获取 lattice 所需的数据:  [(word, start idx, end idx), ...]"
    index = 0
    word_list = []
    for word in pku_seg.cut(text):
        word_len = len(word)
        if word_len > 1 and is_all_chinese(word):
            word_list.append((word, index, index + word_len - 1))
        index += word_len
    return word_list

class InputExample:
    def __init__(self, sentence_id: str, text: str, entities: List[dict] = None):
        self.sentence_id = sentence_id
        self.text = text
        self.entities = entities

    def to_ner_task(self, for_nested_ner: bool = False):    
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        if self.entities is None:
            return self.sentence_id, self.text
        else:
            if not for_nested_ner:
                label = [NO_ENT] * len(self.text)
            else:
                label1 = [NO_ENT] * len(self.text)
                label2 = [NO_ENT] * len(self.text)

            def _write_label(_label: list, _type: str, _start: int, _end: int):
                for i in range(_start, _end + 1):
                    if i == _start:
                        _label[i] = f"B-{_type}"
                    else:
                        _label[i] = f"I-{_type}"

            for entity in self.entities:
                start_idx = entity["start_idx"]
                end_idx = entity["end_idx"]
                entity_type = entity["type"]
                assert entity["entity"] == self.text[start_idx: end_idx + 1], f"{entity} mismatch: `{self.text}`"

                if not for_nested_ner:
                    _write_label(label, entity_type, start_idx, end_idx)
                else:
                    if entity_type=='sym':
                         _write_label(label2, entity_type, start_idx, end_idx)
                    else:
                         _write_label(label1, entity_type, start_idx, end_idx)

                   

            if not for_nested_ner:
                return self.sentence_id, self.text, label
            else:
                return self.sentence_id, self.text, label1, label2


class EEDataloader:
    def __init__(self, cblue_root: str):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")

    @staticmethod
    def _load_json(filename: str) -> List[dict]:
        with open(filename, encoding="utf8") as f:
            return json.load(f)

    @staticmethod
    def _parse(cmeee_data: List[dict]) -> List[InputExample]:
        return [InputExample(sentence_id=str(i), **data) for i, data in enumerate(cmeee_data)]

    def get_data(self, mode: str):
        if mode not in ("train", "dev", "test"):
            raise ValueError(f"Unrecognized mode: {mode} m")
        return self._parse(self._load_json(join(self.data_root, f"CMeEE_{mode}.json")))


class EEDataset(Dataset):
    def __init__(self, cblue_root: str, mode: str, max_length: int, tokenizer, for_nested_ner: bool):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")
        self.max_length = max_length
        self.for_nested_ner = for_nested_ner

        # This flag is used in CRF
        self.no_decode = mode.lower() == "train"

        # Processed data can vary from using different tokenizers
        _tk_class = re.match("<class '.*\.(.*)'>", str(type(tokenizer))).group(1)
        _head = 2 if for_nested_ner else 1
        cache_file = join(self.data_root, f"cache_{mode}_{max_length}_{_tk_class}_{_head}head.pkl")

        if False and exists(cache_file):
            with open(cache_file, "rb") as f:
                self.examples, self.data = pickle.load(f)
            logger.info(f"Load cached data from {cache_file}")
        else:
            self.examples = EEDataloader(cblue_root).get_data(mode) # get original data

            self.data= self._preprocess(self.examples, tokenizer) # preprocess
            with open(cache_file, 'wb') as f:
                pickle.dump((self.examples, self.data), f)
            logger.info(f"Cache data to {cache_file}")

    def _preprocess(self, examples: List[InputExample], tokenizer) -> list:
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        is_test = examples[0].entities is None
        data = []
        label2id = EE_label2id
        label2id1=EE_label2id1
        label2id2=EE_label2id2
        word_vector = KeyedVectors.load_word2vec_format('Medical.txt', binary=False)
        word_dict = pickle.load(open('w2v_vocab.pkl', 'rb'))
        word_dict = {word: idx for idx, word in enumerate(word_dict)}
        for example in examples:
            if is_test:
                _sentence_id, text = example.to_ner_task(self.for_nested_ner)
                if not self.for_nested_ner:
                   label = repeat(None, len(text))
                else: 
                   label1 = repeat(None, len(text))
                   label2 = repeat(None, len(text))

            else:
                if not self.for_nested_ner:
                  _sentence_id, text, label = example.to_ner_task(self.for_nested_ner)
                else:
                  _sentence_id, text, label1,label2 = example.to_ner_task(self.for_nested_ner)
            tokens = []
            label_ids = None if is_test else []
            lattice=[]
            lattice = get_lattice_word(text)
            lattice_idx = []  # [start idx，end idx, 词在 word2vec中的index]
            for lword in lattice:
                if (int(lword[2])+2)>=self.max_length: break
                if word_vector.has_index_for(lword[0]):
                    #lword_index = word_dict[lword[0]]
                    lattice_idx.append([lword[1]+1, lword[2]+1, lword[0]])
            if self.for_nested_ner:
               
                label1_ids = None if is_test else []
                label2_ids = None if is_test else []
                for word, L1,L2 in zip(text, label1, label2):
                    token = tokenizer.tokenize(word)
                    if not token:
                        token = [tokenizer.unk_token]
                    tokens.extend(token)

                    if not is_test:
                        label1_ids.extend([label2id1[L1]] + [tokenizer.pad_token_id] * (len(token) - 1))
                
                        label2_ids.extend([label2id2[L2]] + [tokenizer.pad_token_id] * (len(token) - 1))
            else:
                for word, L in zip(text, label):
                    token = tokenizer.tokenize(word)
                    if not token:
                        token = [tokenizer.unk_token]
                    tokens.extend(token)

                    if not is_test:
                        label_ids.extend([label2id[L]] + [tokenizer.pad_token_id] * (len(token) - 1))

            tokens = [tokenizer.cls_token] + tokens[: self.max_length - 2] + [tokenizer.sep_token]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            #print('text:',text)
            #print('tokens:',tokens)
            #print('token_ids:',token_ids)
            #print('lattice:',lattice)
            #print('lattice_idx:',lattice_idx)
            if not is_test:
                if not self.for_nested_ner:
                   label_ids = [label2id[NO_ENT]] + label_ids[: self.max_length - 2] + [label2id[NO_ENT]]
                 
                   data.append((token_ids,(len(text)+2),lattice_idx,label_ids))
                else:
                    label1_ids = [label2id1[NO_ENT]] + label1_ids[: self.max_length - 2] + [label2id1[NO_ENT]]
                    label2_ids = [label2id2[NO_ENT]] + label2_ids[: self.max_length - 2] + [label2id2[NO_ENT]]
                    data.append((token_ids,(len(text)+2),lattice_idx,label1_ids,label2_ids))
            else:
                data.append((token_ids,(len(text)+2),lattice_idx,))
        


        return data

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.no_decode


class CollateFnForEE:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = EE_label2id[NER_PAD], for_nested_ner: bool = False):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.for_nested_ner = for_nested_ner
    def __call__(self, batch) -> dict:
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        inputs = [x[0] for x in batch]
        no_decode_flag = batch[0][1]
        #char_len=[x[2] for x in batch]
       
        input_ids = [x[0]  for x in inputs]
        char_len=[x[1] for x in inputs]
        lattice = [x[2] for x in inputs]
        #word_index=[x[2] for x in lattice]
        #word_start_idx=[x[0] for x in lattice]
        #word_end_idx=[x[1] for x in lattice]
        #print('word_index:',word_index)
        #print('start:',word_start_idx)
        #print('end:',word_end_idx)
        if not self.for_nested_ner:
            labels    = [x[3]  for x in inputs] if len(inputs[0]) > 3 else None
        else:
            labels    = [x[3]  for x in inputs] if len(inputs[0]) > 3 else None
            labels2   = [x[4]  for x in inputs] if len(inputs[0]) > 3 else None
        
    
        max_len = max(map(len, input_ids))
        max_word_len=max(map(len, lattice))
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, _ids in enumerate(input_ids):
            attention_mask[i][:len(_ids)] = 1
            _delta_len = max_len - len(_ids)
            input_ids[i] += [self.pad_token_id] * _delta_len
            
            if labels is not None:
                labels[i] += [self.label_pad_token_id] * _delta_len
            if self.for_nested_ner:
                if labels2 is not None:
                    labels2[i] += [self.label_pad_token_id] * _delta_len

        if not self.for_nested_ner:
            inputs = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": attention_mask,
                "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
                "no_decode": no_decode_flag,
                "char_lens": char_len,
                "lattice": lattice
            }
        else:
            inputs = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": attention_mask,
                "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None, # modify this
                "labels2": torch.tensor(labels2, dtype=torch.long) if labels2 is not None else None, # modify this
                "no_decode": no_decode_flag,
                "char_lens": char_len,
                "lattice": lattice
            }

        return inputs


if __name__ == '__main__':
    import os
    from os.path import expanduser
    from transformers import BertTokenizer

    
    MODEL_NAME = "../bert-base-chinese"
    CBLUE_ROOT = "../data/CBLUEDatasets"

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    dataset = EEDataset(CBLUE_ROOT, mode="train", max_length=10000, tokenizer=tokenizer, for_nested_ner=False)

    batch = [dataset[0], dataset[1], dataset[2]]
    inputs = CollateFnForEE(pad_token_id=tokenizer.pad_token_id, for_nested_ner=False)(batch)
    print(inputs)

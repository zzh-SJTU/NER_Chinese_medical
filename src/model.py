
from audioop import bias
from typing import Optional
from unicodedata import bidirectional

import torch
from torchcrf import CRF
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers.file_utils import ModelOutput
from torch.nn import functional as F
from ee_data import EE_label2id1, NER_PAD
import pickle
import gensim
from gensim.models import KeyedVectors,word2vec
from flat import FLAT
NER_PAD_ID = EE_label2id1[NER_PAD]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#device = torch.device("cpu")
@dataclass
class NEROutputs(ModelOutput):
    """
    NOTE: `logits` here is the CRF decoding result.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.LongTensor] = None


class LinearClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.num_labels = num_labels
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        self.loss_fct = CrossEntropyLoss()

    def _pred_labels(self, _logits):
        return torch.argmax(_logits, dim=-1)

    def forward(self, hidden_states, labels=None, no_decode=False):
        _logits = self.layers(hidden_states)
        loss, pred_labels = None, None

        if labels is None:
            pred_labels = self._pred_labels(_logits)    
        else:
            loss = self.loss_fct(_logits.view(-1, self.num_labels), labels.view(-1))
            if not no_decode:
                pred_labels = self._pred_labels(_logits)
 
        return NEROutputs(loss, pred_labels)


class CRFClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()

        self.num_labels = num_labels
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        self.crf=CRF(self.num_labels,batch_first=True)
        '''NOTE: This is where to modify for CRF.

        '''
        # output = self.classifier.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
    def _pred_labels(self,emissions,attention_mask,label_pad_token_id):
        pred_labels = self.crf.decode(emissions, attention_mask.bool())
        pred_labels_pad = [F.pad(torch.tensor(text),(0,attention_mask.shape[1]-len(text)),'constant',label_pad_token_id) for text in pred_labels]
        return torch.stack(pred_labels_pad).long()

    def forward(self, hidden_states, attention_mask, labels=None, no_decode=False, label_pad_token_id=NER_PAD_ID):    
       
        loss, pred_labels = None, None
        emissions=self.layers(hidden_states)
        if labels is None:
            pred_labels = self._pred_labels(emissions,attention_mask,label_pad_token_id)    
        else:
            loss = -self.crf(emissions = emissions, tags=labels, mask=attention_mask.bool())
            if  no_decode:
                pred_labels = self._pred_labels(emissions,attention_mask,label_pad_token_id)
        return NEROutputs(loss, pred_labels)


def _group_ner_outputs(output1: NEROutputs, output2: NEROutputs):
    """ logits: [batch_size, seq_len] ==> [batch_size, seq_len, 2] """
    grouped_loss, grouped_logits = None, None

    if not (output1.loss is None or output2.loss is None):
        grouped_loss = (output1.loss + output2.loss) / 2

    if not (output1.logits is None or output2.logits is None):
        grouped_logits = torch.stack([output1.logits, output2.logits], dim=-1)

    return NEROutputs(grouped_loss, grouped_logits)


class BertForLinearHeadNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)

        #self.classifier = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.classifier = LinearClassifier(256, num_labels1, config.hidden_dropout_prob)
        self.w2v_linear = nn.Linear(300, 768)

        self.layer_norm = nn.LayerNorm(768, eps=1e-12)

        self.dropout = nn.Dropout(0.1)

        self.w2v_array = pickle.load(open('w2v_vector.pkl', 'rb'))

        self.flat = FLAT()

        self.init_weights()

        
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
            char_lens=None,
            lattice=None,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        
        #+print('input_ids:',input_ids.shape)
        char_vec = sequence_output
        #for vec in char_vec:
        #        vec.detach()
        # 当使用预训练 word vec 时，载入静态参数。  另外可以尝试输入 id，构建embedding重新训练参数
        # 构建 FLAT 的输入格式，这部分时组织输入，不进入计算图。
        char_word_vec = []
        #print('char_vec shape:',char_vec.shape)
        #print('char_lens len:',len(char_lens))
        #print('char_len:',char_lens)
        max_word_len=max(map(len, lattice))
        #print('max_word_len:',max_word_len)
        max_len=char_vec.shape[1]+max_word_len
        #print('max_len:',max_len)
        char_word_mask = torch.zeros((8, max_len), dtype=torch.long)
        pos = torch.arange(0, char_vec.size(1)).long().unsqueeze(dim=0).to(device)
        pos = pos * attention_mask.long()

        pad = torch.tensor([0 for _ in range(max_len - attention_mask.size(1))]).unsqueeze(0).repeat(attention_mask.size(0), 1).to(device)
        pos = torch.cat((pos, pad), dim=1)
        char_word_head = pos.clone()
        char_word_tail = pos.clone()
        for i, bchar_vec in enumerate(char_vec):
            #print('len_bchar_vec:',len(bchar_vec))
            #rint('char_len:',char_lens[i])
            bert_vec = []
            word_vec = []
            pad_vec = []
            for idx, vec in enumerate(bchar_vec):
                if idx < char_lens[i]:
                    bert_vec.append(vec)
            lattice_per_batch=lattice[i]
            #print('lattice_per_batch:',lattice_per_batch)
            for idx, word in enumerate(lattice_per_batch):
                word_idx=word[2]
                #print('word_idx:',word_idx)
                #print('word:',word)
                vec = torch.tensor(self.w2v_array[int(word_idx)]).float().to(device)
                word_vec.append(vec)
                char_word_head[i][len(bert_vec)+idx]=word[0]
                char_word_tail[i][len(bert_vec)+idx]=word[1]
            bert_vec = torch.stack(bert_vec, dim=0).to(device)
            #print('bert_vec:',bert_vec.shape)
            #print('pos:',pos[i])
            #print('char_word_head:',char_word_head[i])
            #print('char_word_tail:',char_word_tail[i])
            if len(word_vec) > 0:
                word_vec = torch.stack(word_vec, dim=0).to(device)  # 2维
                word_vec = self.w2v_linear(word_vec)
                #print('word_vec:',word_vec.shape)
                new_vec = torch.cat((bert_vec, word_vec), dim=0)  # 2维
            else:
                new_vec = bert_vec  # 2维
            new_vec = self.layer_norm(new_vec)
            new_vec = self.dropout(new_vec)
            char_word_mask[i][:new_vec.shape[0]] = 1
            pad_len=max_len-new_vec.shape[0]
            #print('new_vec:',new_vec.shape)
            #print('pad_len:',pad_len)
            for pad in range(pad_len):
                pad_vec.append(torch.zeros(768).to(device))
            if len(pad_vec) > 1:
                pad_vec = torch.stack(pad_vec, dim=0).to(device) # 2维
                new_vec = torch.cat((new_vec, pad_vec), dim=0)
            elif len(pad_vec) == 1:
                pad_vec = pad_vec[0].unsqueeze(0)
                new_vec = torch.cat((new_vec, pad_vec), dim=0)
            #print('new_vec:',new_vec.shape)
            #print('mask:',char_word__mask[i].shape)
            char_word_vec.append(new_vec)
        char_word_vec = torch.stack(char_word_vec, dim=0).float().to(device)
        
        #print('char_Word_Vec:',char_word_vec.shape)
        #print('char_word_head:',char_word_head.shape)
        #print('char_word_tail:',char_word_tail.shape)
        #print('char_word_mask:',char_word_mask.shape)
        #print('char_len:',attention_mask.size(1))
        #last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        #  — Sequence of hidden-states at the output of the last layer of the model.
        encoder_outputs={'char_word_vec': char_word_vec,
                    'char_word_mask': char_word_mask.to(device).bool(),
                    'char_word_s': char_word_head,
                    'char_word_e': char_word_tail,
                    'char_len': attention_mask.size(1)}
        fusion_outputs = self.flat(encoder_outputs)
        #print('sequence_outputs:',sequence_output.shape)
        #print('fusion_outputs:',fusion_outputs.shape)
        output = self.classifier.forward(fusion_outputs, labels, no_decode=no_decode)
        return output


class BertForLinearHeadNestedNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        '''NOTE: This is where to modify for Nested NER.
        
        '''
        self.w2v_linear = nn.Linear(512, 1024)

        self.layer_norm = nn.LayerNorm(1024, eps=1e-12)

        self.dropout = nn.Dropout(0.1)

        self.w2v_array = KeyedVectors.load_word2vec_format('Medical.txt', binary=False)

        self.flat = FLAT()
        self.classifier = LinearClassifier(256, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = LinearClassifier(256, num_labels2, config.hidden_dropout_prob)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
            char_lens=None,
            lattice=None,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        char_vec = sequence_output
        for vec in char_vec:
                vec.detach()
        # 当使用预训练 word vec 时，载入静态参数。  另外可以尝试输入 id，构建embedding重新训练参数
        # 构建 FLAT 的输入格式，这部分时组织输入，不进入计算图。
        char_word_vec = []
        #print('char_vec shape:',char_vec.shape)
        #print('char_lens len:',len(char_lens))
        #print('char_len:',char_lens)
        max_word_len=max(map(len, lattice))
        #print('max_word_len:',max_word_len)
        max_len=char_vec.shape[1]+max_word_len
        #print('max_len:',max_len)
        char_word_mask = torch.zeros((4, max_len), dtype=torch.long)
        pos = torch.arange(0, char_vec.size(1)).long().unsqueeze(dim=0).to(device)
        pos = pos * attention_mask.long()

        pad = torch.tensor([0 for _ in range(max_len - attention_mask.size(1))]).unsqueeze(0).repeat(attention_mask.size(0), 1).to(device)
        pos = torch.cat((pos, pad), dim=1)
        char_word_head = pos.clone()
        char_word_tail = pos.clone()
        for i, bchar_vec in enumerate(char_vec):
            #print('len_bchar_vec:',len(bchar_vec))
            #rint('char_len:',char_lens[i])
            bert_vec = []
            word_vec = []
            pad_vec = []
            for idx, vec in enumerate(bchar_vec):
                if idx < char_lens[i]:
                    bert_vec.append(vec)
            lattice_per_batch=lattice[i]
            #print('lattice_per_batch:',lattice_per_batch)
            for idx, word in enumerate(lattice_per_batch):
                #word_idx=word[2]
                word_token = word[2]
                #print('word_idx:',word_idx)
                #print('word:',word)
                vec = torch.tensor(self.w2v_array.get_vector(word_token)).float().to(device)
                word_vec.append(vec)
                char_word_head[i][len(bert_vec)+idx]=word[0]
                char_word_tail[i][len(bert_vec)+idx]=word[1]
            bert_vec = torch.stack(bert_vec, dim=0).to(device)
            #print('bert_vec:',bert_vec.shape)
            #print('pos:',pos[i])
            #print('char_word_head:',char_word_head[i])
            #print('char_word_tail:',char_word_tail[i])
            if len(word_vec) > 0:
                word_vec = torch.stack(word_vec, dim=0).to(device)  # 2维
                word_vec = self.w2v_linear(word_vec)
                #print('word_vec:',word_vec.shape)
                new_vec = torch.cat((bert_vec, word_vec), dim=0)  # 2维
            else:
                new_vec = bert_vec  # 2维
            new_vec = self.layer_norm(new_vec)
            new_vec = self.dropout(new_vec)
            char_word_mask[i][:new_vec.shape[0]] = 1
            pad_len=max_len-new_vec.shape[0]
            #print('new_vec:',new_vec.shape)
            #print('pad_len:',pad_len)
            for pad in range(pad_len):
                pad_vec.append(torch.zeros(1024).to(device))
            if len(pad_vec) > 1:
                pad_vec = torch.stack(pad_vec, dim=0).to(device) # 2维
                new_vec = torch.cat((new_vec, pad_vec), dim=0)
            elif len(pad_vec) == 1:
                pad_vec = pad_vec[0].unsqueeze(0)
                new_vec = torch.cat((new_vec, pad_vec), dim=0)
            #print('new_vec:',new_vec.shape)
            #print('mask:',char_word__mask[i].shape)
            char_word_vec.append(new_vec)
        char_word_vec = torch.stack(char_word_vec, dim=0).float().to(device)
        
        encoder_outputs={'char_word_vec': char_word_vec,
                    'char_word_mask': char_word_mask.to(device).bool(),
                    'char_word_s': char_word_head,
                    'char_word_e': char_word_tail,
                    'char_len': attention_mask.size(1)}
        fusion_outputs = self.flat(encoder_outputs)
        output1 = self.classifier.forward(fusion_outputs, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(fusion_outputs, labels2, no_decode=no_decode)
        output=_group_ner_outputs(output1,output2)
        return output
      


class BertForCRFHeadNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        self.classifier = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output = self.classifier.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        
        return output


   
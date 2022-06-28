
import numpy as np

from typing import List, Union, NamedTuple, Tuple, Counter

from ee_data import EE_label2id, EE_label2id1, EE_label2id2, EE_id2label1, EE_id2label2, EE_id2label, NER_PAD, _LABEL_RANK,NO_ENT


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class ComputeMetricsForNER: # training_args  `--label_names labels `
    def __call__(self, eval_pred) -> dict:
        predictions, labels = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels[labels == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        #'''NOTE: You need to finish the code of computing f1-score.
        entities_predict = extract_entities(predictions)
        entities_gt = extract_entities(labels)
        pred_true = 0
        pred = 0
        true = 0
        for batch_id in range(len(entities_gt)):
            batch_pre = entities_predict[batch_id]
            batch_gt = entities_gt[batch_id]
            for entity in batch_pre:
                if entity in batch_gt:
                    pred_true += 1
            pred += len(batch_pre)
            true += len(batch_gt)
        #'''

        return { "f1": 2* pred_true/(pred+true)}


class ComputeMetricsForNestedNER: # training_args  `--label_names labels labels2`
    def __call__(self, eval_pred) -> dict:
        predictions, (labels1, labels2) = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len, 2]
        labels1[labels1 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels2[labels2 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        pre_entity1 = extract_entities(predictions[:,:,0],True,True)
        pre_entity2 = extract_entities(predictions[:,:,1],True,False)
        gt_entity1 = extract_entities(labels1,True,True)
        gt_entity2 = extract_entities(labels2,True,False)
        pred_true = 0
        pred = 0
        true = 0
        for batch_id in range(len(gt_entity1)):
            batch_pre1 = pre_entity1[batch_id]
            batch_pre2 = pre_entity2[batch_id]
            batch_gt1 = gt_entity1[batch_id]
            batch_gt2 = gt_entity2[batch_id]
            for entity in batch_pre1:
                if entity in batch_gt1:
                    pred_true+=1
            for entity in batch_pre2:
                if entity in batch_gt2:
                    pred_true+=1
            pred+=len(batch_pre1)+len(batch_pre2)
            true+=len(batch_gt1) + len(batch_gt2)
        # '''NOTE: You need to finish the code of computing f1-score.

        # '''

        return { "f1": 2* pred_true/(pred+true) }


def extract_entities(batch_labels_or_preds: np.ndarray, for_nested_ner: bool = False, first_labels: bool = True) -> List[List[tuple]]:
    """
    本评测任务采用严格 Micro-F1作为主评测指标, 即要求预测出的 实体的起始、结束下标，实体类型精准匹配才算预测正确。
    
    Args:
        batch_labels_or_preds: The labels ids or predicted label ids.  
        for_nested_ner:        Whether the input label ids is about Nested NER. 
        first_labels:          Which kind of labels for NestNER.
    """
    batch_labels_or_preds[batch_labels_or_preds == -100] = EE_label2id1[NER_PAD]  # [batch, seq_len]
    
    if not for_nested_ner:
        id2label = EE_id2label
    else:
        id2label = EE_id2label1 if first_labels else EE_id2label2
    batch_entities = []  # List[List[(start_idx, end_idx, type)]]
    for i in range(batch_labels_or_preds.shape[0]):
        one_batch = []
        current_entity = []
        for j in range(batch_labels_or_preds.shape[1]):
            idx = batch_labels_or_preds[i][j]
            if id2label[idx] != NER_PAD and id2label[idx] != NO_ENT:
                label = id2label[idx]
                if label[0] == 'B':  
                    if len(current_entity) == 0:
                        current_entity = [j,j,[label]]
                    else:
                        dic_num = {}
                        for label_1 in current_entity[2]:
                            label_type = label_1[2:]
                            if label_type in dic_num.keys():
                                dic_num[label_type] += 1
                            else:
                                dic_num[label_type] = 1
                        sorted_dic = sorted(dic_num.items(),key=lambda v: v[1],reverse=True)
                        freq_max = sorted_dic[0][1]
                        list_most_label = []
                        for pair in sorted_dic:
                            if pair[1] == freq_max:
                                list_most_label.append(pair[0])
                            else:
                                break
                        if len(list_most_label) == 1:
                            one_batch.append((current_entity[0],current_entity[1],list_most_label[0]))
                        else:
                            freq_rank = -1e10
                            for label_2 in list_most_label:
                                if _LABEL_RANK[label_2] > freq_rank:
                                    freq_rank = _LABEL_RANK[label_2]
                                    final_label = label_2
                            one_batch.append((current_entity[0],current_entity[1],final_label))
                        current_entity = [j,j,[label]]                       
                else:
                    if len(current_entity) != 0:
                        current_entity[1] = j
                        current_entity[2].append(label)
                '''
                if len(current_entity) == 0:  
                    label = id2label[idx]
                    current_entity = [j,j,[label]]
                else:
                    label = id2label[idx]
                    current_entity[1] = j
                    current_entity[2].append(label)
                '''
            else:
                if len(current_entity) != 0:
                    dic_num = {}
                    for label_1 in current_entity[2]:
                        label_type = label_1[2:]
                        if label_type in dic_num.keys():
                            dic_num[label_type] += 1
                        else:
                            dic_num[label_type] = 1
                    sorted_dic = sorted(dic_num.items(),key=lambda v: v[1],reverse=True)
                    freq_max = sorted_dic[0][1]
                    list_most_label = []
                    for pair in sorted_dic:
                        if pair[1] == freq_max:
                            list_most_label.append(pair[0])
                        else:
                            break
                    if len(list_most_label) == 1:
                        one_batch.append((current_entity[0],current_entity[1],list_most_label[0]))
                    else:
                        freq_rank = -1e10
                        for label_2 in list_most_label:
                            if _LABEL_RANK[label_2] > freq_rank:
                                freq_rank = _LABEL_RANK[label_2]
                                final_label = label_2
                        one_batch.append((current_entity[0],current_entity[1],final_label))
                    current_entity = []
        batch_entities.append(one_batch)
                                
    # '''NOTE: You need to finish this function of extracting entities for generating results and computing metrics.
    
    # '''
    return batch_entities


if __name__ == '__main__':

    # Test for ComputeMetricsForNER
    predictions = np.load('../test_files/predictions.npy')
    labels = np.load('../test_files/labels.npy')

    metrics = ComputeMetricsForNER()(EvalPrediction(predictions, labels))
    if abs(metrics['f1'] - 0.606179116) < 1e-5:
        print('You passed the test for ComputeMetricsForNER.')
    else:
        print('The result of ComputeMetricsForNER is not right.')
    
    # Test for ComputeMetricsForNestedNER
    predictions = np.load('../test_files/predictions_nested.npy')
    labels1 = np.load('../test_files/labels1_nested.npy')
    labels2 = np.load('../test_files/labels2_nested.npy')

    metrics = ComputeMetricsForNestedNER()(EvalPrediction(predictions, (labels1, labels2)))

    if abs(metrics['f1'] - 0.60333644) < 1e-5:
        print('You passed the test for ComputeMetricsForNestedNER.')
    else:
        print('The result of ComputeMetricsForNestedNER is not right.')
    
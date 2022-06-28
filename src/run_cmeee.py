
import json
from lib2to3.pgen2.token import NOTEQUAL
from os.path import join
from typing import List
import os
from sklearn.metrics import precision_recall_fscore_support
from transformers.models.bert.modeling_bert import BertEmbeddings, BertAttention
from transformers import set_seed, BertTokenizer, Trainer, HfArgumentParser, TrainingArguments, BertLayer
from args import ModelConstructArgs, CBLUEDataArgs
from logger import get_logger
from transformers import  AdamW
from ee_data import EE_label2id2, EEDataset, EE_NUM_LABELS1, EE_NUM_LABELS2, EE_NUM_LABELS, CollateFnForEE, \
    EE_label2id1, NER_PAD, EE_label2id
from model import BertForCRFHeadNER, BertForLinearHeadNER,  BertForLinearHeadNestedNER, CRFClassifier, LinearClassifier
from metrics import ComputeMetricsForNER, ComputeMetricsForNestedNER, extract_entities
from torch.nn import LSTM
import torch
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
MODEL_CLASS = {
    'linear': BertForLinearHeadNER, 
    'linear_nested': BertForLinearHeadNestedNER,
    'crf': BertForCRFHeadNER,
}


class PGD:
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
    def attack(self,
               epsilon=1.,
               alpha=0.3,
               emb_name='emb.',
               is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        # 此处也可以直接用一个成员变量储存 grad，而不用 register_hook 存储在全局变量中
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
        self.grad_backup = {}
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
class CustomTrainer(Trainer):
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        pgd = PGD(model)
        K = 3
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward() 
        pgd.backup_grad()
        for t in range(K):
            pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != K-1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            loss_adv = self.compute_loss(model, inputs)
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        pgd.restore() # 恢复embedding参数
        return loss_adv.detach()
'''
        fgm = FGM(model)
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()
        fgm.attack()
        loss_adv = self.compute_loss(model, inputs)
        loss_adv.backward()
        fgm.restore() 
        return loss_adv.detach()
'''


def get_logger_and_args(logger_name: str, _args: List[str] = None):
    parser = HfArgumentParser([TrainingArguments, ModelConstructArgs, CBLUEDataArgs])
    train_args, model_args, data_args = parser.parse_args_into_dataclasses(_args)

    # ===== Get logger =====
    logger = get_logger(logger_name, exp_dir=train_args.logging_dir, rank=train_args.local_rank)
    for _log_name, _logger in logger.manager.loggerDict.items():
        # 在4.6.0版本的transformers中无效
        if _log_name.startswith("transformers.trainer"):
            # Redirect other loggers' output
            _logger.addHandler(logger.handlers[0])

    logger.info(f"==== Train Arguments ==== {train_args.to_json_string()}")
    logger.info(f"==== Model Arguments ==== {model_args.to_json_string()}")
    logger.info(f"==== Data Arguments ==== {data_args.to_json_string()}")

    return logger, train_args, model_args, data_args


def get_model_with_tokenizer(model_args):
    model_class = MODEL_CLASS[model_args.head_type]

    if 'nested' not in model_args.head_type:
        model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS)
    else:
        model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS1, num_labels2=EE_NUM_LABELS2)
    
    tokenizer = BertTokenizer.from_pretrained(model_args.model_path)
    return model, tokenizer


def generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=False):
    assert len(predictions) == len(test_dataset.examples), \
        f"Length mismatch: predictions({len(predictions)}), test examples({len(test_dataset.examples)})"

    if not for_nested_ner:
        pred_entities1 = extract_entities(predictions[:, 1:], for_nested_ner=False)
        pred_entities2 = [[]] * len(pred_entities1)
    else:
        pred_entities1 = extract_entities(predictions[:, 1:, 0], for_nested_ner=True, first_labels=True)
        pred_entities2 = extract_entities(predictions[:, 1:, 1], for_nested_ner=True, first_labels=False)

    final_answer = []

    for p1, p2, example in zip(pred_entities1, pred_entities2, test_dataset.examples):
        text = example.text
        entities = []
        for start_idx, end_idx, entity_type in p1 + p2:
            entities.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "type": entity_type,
                "entity": text[start_idx: end_idx + 1],
            })
        final_answer.append({"text": text, "entities": entities})

    with open(join(train_args.output_dir, "CMeEE_test.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(f"`CMeEE_test.json` saved")


def main(_args: List[str] = None):
    # ===== Parse arguments =====
    logger, train_args, model_args, data_args = get_logger_and_args(__name__, _args)

    # ===== Set random seed =====
    set_seed(train_args.seed)

    # ===== Get models =====
    model, tokenizer = get_model_with_tokenizer(model_args)
    for_nested_ner = 'nested' in model_args.head_type

    # ===== Get datasets =====
    if train_args.do_train:
        train_dataset = EEDataset(data_args.cblue_root, "train", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner)
        dev_dataset = EEDataset(data_args.cblue_root, "dev", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner)
        logger.info(f"Trainset: {len(train_dataset)} samples")
        logger.info(f"Devset: {len(dev_dataset)} samples")
    else:
        train_dataset = dev_dataset = None

    # ===== Trainer =====
    compute_metrics = ComputeMetricsForNestedNER() if for_nested_ner else ComputeMetricsForNER()
    '''
    layer_names = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        layer_names.append(name)
    lr      = 3e-5
    lr_mult = 0.70

    # placeholder
    parameters      = []
    prev_group_name = layer_names[5].split('.')[3]

    # store params & learning rates
    for idx, name in enumerate(layer_names):
        
        # parameter group name
        if name.split('.')[1] == 'encoder':
            cur_group_name = name.split('.')[3]
            
            # update learning rate
            if cur_group_name != prev_group_name:
                lr *= lr_mult
            prev_group_name = cur_group_name
        
        # display info
        print(f'{idx}: lr = {lr:.6f}, {name}')
        
        # append layer parameters
        parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                        'lr':     lr}]
    optimizer = AdamW(parameters)
    '''
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=CollateFnForEE(tokenizer.pad_token_id, for_nested_ner=for_nested_ner),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
        #optimizers = (optimizer,None)
    )

    if train_args.do_train:
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")

    if train_args.do_predict:
        test_dataset = EEDataset(data_args.cblue_root, "test", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner)
        logger.info(f"Testset: {len(test_dataset)} samples")

        # np.ndarray, None, None
        predictions, _labels, _metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=for_nested_ner)


if __name__ == '__main__':
    main()

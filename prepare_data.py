import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import torch
import nlp
from transformers import T5Tokenizer, BartTokenizer, HfArgumentParser, AutoTokenizer
from google_trans_new import google_translator  
import json
from tqdm import tqdm
from nltk import sent_tokenize
translator = google_translator()

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task: str = field(
        metadata={"help": "Which task 'qa', 'qg', 'e2e_qg', 'ans_ext', 'multi'. 'multi' means 'qa', 'qg', 'ans_ext' tasks"}, 
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    dataset_path: Optional[str] = field(
        default="data/squad_multitask",
        metadata={"help": "Path for dataset directory"}, 
    )
    train_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached train dataset"},
    )
    valid_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached valid dataset"},
    )
    valid_for_qg_only: bool = field(
        default=False,
        metadata={"help": "For multitask dataset valid split should contain only qg task or all tasks."}
    )
    qg_format: Optional[str] = field(
        default='highlight_qg_format',
        metadata={"help": "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"}, 
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )

# tokenizer = T5Tokenizer.from_pretrained("t5-base")
# tokenizer.add_tokens(['<sep>', '<hl>'])

class DataProcessor:
    def __init__(self, tokenizer, model_type="t5", max_source_length=512, max_target_length=32, translate=False):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.translate = translate
        self.hl_token = "<hl>"
        
        if model_type == "t5":
            self.sep_token = "<sep>"
        elif model_type == "bart":
            self.sep_token = "<sep>"
        else:
            self.sep_token = "[SEP]"
  
    def process(self, dataset):
        if self.translate:
            dataset = dataset.map(self._translate_to_vietnamese)
        if self.model_type == "t5":
            dataset = dataset.map(self._add_eos_examples)
        
        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)
        
        return dataset
        
    
    def _translate_to_vietnamese(self, example):
        example['source_text'] = translator.translate(example['source_text'], lang_tgt='vi')
        example['target_text'] = translator.translate(example['target_text'], lang_tgt='vi')
#         print(translator.translate(example['source_text'], dest='vi').text)
        return example

    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example
  
    def _add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)    
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example
  
    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )

        encodings = {
            'source_ids': source_encoding['input_ids'], 
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings


def filter_qa(example):
    return example['task'] == 'qa'

def filter_qg(example):
    return example['task'] == 'qg'

def filter_e2e_qg(example):
    return example['task'] == 'e2e_qg'

def filter_ans_ext(example):
    return example['task'] == 'ans_ext'

def filter_multi(example):
    return example['task'] != 'e2e_qg'


TASK_TO_FILTER_FN = {
    'qa': filter_qa,
    'qg': filter_qg,
    'e2e_qg': filter_e2e_qg,
    'ans_ext': filter_ans_ext,
    'multi': filter_multi
}

def convert_to_squad_format(data_squad):
    source_texts = []
    target_texts = []
    task = []
    for paragraph in tqdm(data_squad):
        for qas in paragraph['qas']:
            question = qas['question']
            answer = qas['answers'][0]['text']
            text = "generate question: " + paragraph['context'].replace(answer, '{hl_token} ' + answer + ' {hl_token}')

            source_texts.append(text)
            target_texts.append(question)
            task.append('qg')
            
        sents = sent_tokenize(paragraph['context'])
        for sent in sents:
            target_answer = ""
            text = "extract answers: " + paragraph['context'].replace(sent, '{hl_token} ' + sent + ' {hl_token}')
            for qas in paragraph['qas']:
                answer = qas['answers'][0]['text']
                if answer in sent:
                    target_answer += answer + " {sep_token}"
            if target_answer != "":
                source_texts.append(text)
                target_texts.append(target_answer)
                task.append('ans_ext')

    data_dict = {'source_text': source_texts,
             'target_text': target_texts,
             'task': task}
    dataset = nlp.Dataset.from_dict(data_dict)
    
    return dataset

def main():
    parser = HfArgumentParser((DataTrainingArguments,))

    data_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if data_args.model_type == 't5':
        # tokenizer = T5Tokenizer.from_pretrained("t5-base")
        # tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-vi-en-small")
    else:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    tokenizer.add_tokens(['<sep>', '<hl>'])
    
    with open ("../../Nanibot_ZaloAIChallenge2019_VietnameseWikiQA/Dataset/squad_vi.json", 'r', encoding='utf-8') as f:
        squad_vi = json.load(f)

    data_squad = squad_vi['data'][0]['paragraphs']

    train_squad = data_squad[:17000]
    valid_squad = data_squad[17000:]
    
    train_dataset = convert_to_squad_format(train_squad)
    valid_dataset = convert_to_squad_format(valid_squad)
    # train_dataset = nlp.load_dataset(data_args.dataset_path, name=data_args.qg_format, split=nlp.Split.TRAIN)
    # valid_dataset = nlp.load_dataset(data_args.dataset_path, name=data_args.qg_format, split=nlp.Split.VALIDATION)

    processor = DataProcessor(
        tokenizer,
        model_type=data_args.model_type,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        translate=False
    )

    train_dataset = train_dataset.filter(TASK_TO_FILTER_FN[data_args.task])
    if data_args.task == 'multi' and data_args.valid_for_qg_only:
        logger.info("processing valid data only for qg task")
        valid_dataset = valid_dataset.filter(filter_qg)
    else:
        valid_dataset = valid_dataset.filter(TASK_TO_FILTER_FN[data_args.task])

    
    train_dataset = processor.process(train_dataset)
    valid_dataset = processor.process(valid_dataset)

    columns = ["source_ids", "target_ids", "attention_mask"]
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    if data_args.train_file_name is None:
        train_file_name = f"train_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt"
        train_path = os.path.join("data", train_file_name)

        valid_file_name = f"valid_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt"
        valid_path = os.path.join("data", valid_file_name)
    else:
        train_path = os.path.join("data", data_args.train_file_name)
        valid_path = os.path.join("data", data_args.valid_file_name)
    
    torch.save(train_dataset, train_path)
    logger.info(f"saved train dataset at {train_path}")
    
    torch.save(valid_dataset, valid_path)
    logger.info(f"saved validation dataset at {valid_path}")
    
    tokenizer_path = f"{data_args.model_type}_qg_tokenizer"
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"saved tokenizer at {tokenizer_path}")


if __name__ == "__main__":
    main()

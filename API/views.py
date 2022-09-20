from django.shortcuts import render
import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification
from rest_framework.decorators import api_view
from rest_framework.response import Response

from utils import init_logger, load_tokenizer

logger = logging.getLogger(__name__)


args = torch.load(os.path.join("./model", 'training_args.bin'))
device = "cpu"
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)  # Config will be automatically loaded from model_dir
model.to(device)
model.eval()



# line은 문장이 들어오면 됨 
def convert_input_file_to_tensor_dataset(line,
                                         args,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    tokenizer = load_tokenizer(args)

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []

   
    line = line.strip()
    tokens = tokenizer.tokenize(line)
    # Account for [CLS] and [SEP]
    special_tokens_count = 2
    if len(tokens) > args.max_seq_len - special_tokens_count:
        tokens = tokens[:(args.max_seq_len - special_tokens_count)]

    # Add [SEP] token
    tokens += [sep_token]
    token_type_ids = [sequence_a_segment_id] * len(tokens)

    # Add [CLS] token
    tokens = [cls_token] + tokens
    token_type_ids = [cls_token_segment_id] + token_type_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = args.max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    all_input_ids.append(input_ids)
    all_attention_mask.append(attention_mask)
    all_token_type_ids.append(token_type_ids)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

    return dataset


def predict(pred_config):
    # load model and args

    logger.info(args)

    # Convert input file to TensorDataset
    dataset = convert_input_file_to_tensor_dataset(pred_config, args)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)

    # Write to output file
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for pred in preds:
            f.write("{}\n".format(pred))

    logger.info("Prediction Done!")
    
    

@api_view(["POST"])
def predict_model(request):
    predict_sentence = request.data.get('text')
    print(predict_sentence)
    print(args)
    dataset = convert_input_file_to_tensor_dataset(predict_sentence, args)
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    preds = int(np.argmax(preds, axis=1))

    
    return_value = { "sentence" : preds}
    
    
    return Response(return_value)

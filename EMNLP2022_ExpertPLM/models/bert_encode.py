# -*- coding: utf-8 -*-

'''
BERT Encode
--------------------------------------
Input: sequence list
Output: torch.tensor for sequence list
--------------------------------------
'''

import torch
import transformers
import pytorch_pretrained_bert

import numpy as np

#  voc_ = '../bert-pytorch/bert-base-uncased-vocab.txt'
#  model_path = '../bert-pytorch/bert-base-uncased'
voc_ = 'bert-base-uncased'
model_path = 'bert-base-uncased'

# --------BERT-------------


class Bert_encoder():

    def __init__(self, voc=voc_, model_path=model_path):

        self.tokenizer = transformers.BertTokenizer.from_pretrained(voc)
        self.model = pytorch_pretrained_bert.BertModel.from_pretrained(model_path)

    def token_encode(self, str_list):
        '''
        1.Tokenize the sequence
        2.Encode the sequence
        '''
        str_list = self.tokenizer(str_list, padding='longest', return_tensors="pt")
        return str_list['input_ids'], str_list['token_type_ids'], str_list['attention_mask']

    def bert_embedding(self, str_list):
        '''
        ---------------------------------------------------------------
        Input:sequence(B*sequences)
        Output:sequence embedding, sequence pool embedding (B*L*H, B*H)
        ---------------------------------------------------------------
        B:Batch_size
        L:The Longest length of the sequence
        H:Hidden Features
        ---------------------------------------------------------------
        '''
        ids, type_ids, mask = self.token_encode(str_list)
        title_embedding = self.model(ids, type_ids, mask, output_all_encoded_layers=False)

        return title_embedding[0].cuda(), title_embedding[1].cuda()



# -*- coding: utf-8 -*-
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .bert_encode import Bert_encoder
import numpy as np
from pytorch_pretrained_bert import BertModel
import transformers


class Bert_base(nn.Module):
    '''
    Bert Baseline
    '''
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.dropout = nn.Dropout(opt.drop_out)

        self.f_fc = nn.Linear(768 * 2, opt.fea_size)
        self.predict = nn.Linear(opt.fea_size, 1)

        self.bert_encoder_a = BertModel.from_pretrained('./EMNLP2022_ExpertPLM/bert_pretrain/checkpoints/mlm_user_modeling/')
        self.bert_encoder_b = BertModel.from_pretrained('./EMNLP2022_ExpertPLM/bert_pretrain/checkpoints/mlm_user_modeling/')

        self.reset_para()

    def forward(self, data):

        a_id, bid_input, bid_type, batt_mask, qid_input, qid_type, qatt_mask = data
        
        # two tower
        q_title_r = self.bert_encoder_b(bid_input, batt_mask, output_all_encoded_layers=False)[1]
        aid_q_title_r = self.bert_encoder_a(qid_input, qatt_mask, output_all_encoded_layers=False)[1]

        # single tower
        # int_embedding = self.bert_encoder(bid_input, batt_mask, output_all_encoded_layers=False)[1]
        
        # -----------------------------embedding---------------------------

        f = self.f_fc(self.dropout(torch.cat([q_title_r, aid_q_title_r], 1)))
        # f = self.f_fc(self.dropout(int_embedding))
        out = self.predict(F.relu(f))

        return out

    def reset_para(self):
        ''' 参数初始化 '''

        fcs = [self.f_fc, self.predict]
        for fc in fcs:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.uniform_(fc.bias, 0.01)

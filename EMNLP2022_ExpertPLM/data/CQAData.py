# -*- coding: utf-8 -*-
import ipdb
import transformers

from torch.utils.data import Dataset
import numpy as np

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

def token_encode(str_list):
    '''
    1.Tokenize the sequence
    2.Encode the sequence
    '''
    str_list = tokenizer.tokenize(str_list)
    index_tokens = tokenizer.convert_tokens_to_ids(str_list)
    return [index_tokens]


class CQAData(Dataset):

    def __init__(self, dataset, mode):

        self.a_qid = np.load(f"{dataset}/a_qid.npy", allow_pickle=True).tolist()
        self.q_title = np.load(f'{dataset}/q_title.npy', allow_pickle=True).tolist()
        self.q_body = np.load(f'{dataset}/q_body.npy', allow_pickle=True).tolist()
        self.q_tag = np.load(f'{dataset}/q_tag.npy', allow_pickle=True).tolist()

        if mode == 'Train':
            path = f'{dataset}/train'
        elif mode == 'Test':
            path = f'{dataset}/test'
        elif mode == 'Dev':
            path = f'{dataset}/dev'

        self.qid_list = np.load(f'{path}/qid_list.npy', allow_pickle=True).tolist()
        self.aid_list = np.load(f'{path}/aid_list.npy', allow_pickle=True).tolist()
        self.label_list = np.load(f'{path}/label_list.npy', allow_pickle=True).tolist()

    def __getitem__(self, idx):
        assert idx < len(self)

        q_id = self.qid_list[idx]
        a_id = self.aid_list[idx]

        label = self.label_list[idx]

        q_body = self.q_body[q_id]
        q_title = self.q_title[q_id]
        q_tag = self.q_tag[q_id]

        aid_qid_list = self.a_qid[a_id]

        if q_id in aid_qid_list:
            aid_qid_list.remove(q_id)

        aid_q_titles = [self.q_title[q] for q in aid_qid_list] # answerer id 对应的历史回答的 问题的title
        aid_q_bodys = [self.q_body[q] for q in aid_qid_list] # answerer id 对应的历史回答的 问题的body
        aid_q_tags = [self.q_tag[q] for q in aid_qid_list] # answerer id 对应的历史回答的 问题的tag

        # bert_title = q_title + '[SEP]'
        # back = ' '
        # back = back.join(aid_q_titles)
        # bert_title = bert_title + back
        bert_title = f"{q_title} [SEP] {' '.join(aid_q_titles)}"

        # str_list = tokenizer(bert_title, padding="max_length", truncation=True, max_length=512)
        # bid_input = str_list['input_ids']
        # btype_input = str_list['token_type_ids']
        # baat_mask = str_list['attention_mask']

        # x = [label, q_id, a_id, q_title, q_body, q_tag, \
            # aid_q_titles, aid_q_bodys, aid_q_tags, bid_input, btype_input, baat_mask] # label questionid title 某用户历史回答问题title 只有accepted answerer为label=1
        x = [label, q_id, a_id, q_title, q_body, q_tag, \
            aid_q_titles, aid_q_bodys, aid_q_tags, bert_title] # label questionid title 某用户历史回答问题title 只有accepted answerer为label=1
        return x

    def __len__(self):
        return len(self.label_list)


# -*- coding: utf-8 -*-

import numpy as np
from accelerate import Accelerator

accelerator = Accelerator()

class DefaultConfig:

    model = 'Bert_base'
    dataset = 'print'

    # -------------base config-----------------------#
    gpu_id = 1
    gpu_ids = []

    seed = 2019
    num_epochs = 1
    num_workers = 0

    enc_method = 'transformer'

    optimizer = 'Adam'
    weight_decay = 5e-4  # optimizer rameteri
    lr = 2e-5
    drop_out = 0.2

    metrics = ['mean_mrr', 'P@1;3', 'ndcg@20']

    CANDIDATE_NUM = 20
    word_dim = 100
    fea_size = 100
    vocab_size = 30000

    batch_size = 4

    def set_path(self, name):
        '''
        specific
        '''
        self.data_path = f'./data/{name}'

        self.answerer_id_path = f'{self.data_path}/aid.npy'
        self.answerer_history_path = f'{self.data_path}/a_history.npy'

    def parse(self, kwargs):

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        accelerator.print('*************************************************')
        accelerator.print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))

        accelerator.print('*************************************************')


class Codegolf_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Codegolf')
    dataset = 'Codegolf'
    a_num = 2874
    q_num = 5101
    tag_num = 258

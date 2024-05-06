# -*- encoding: utf-8 -*-
import time
import random
import fire

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import CQAData
import config
from utils import cal_metric, group_resuts
import models
from transformers import BertTokenizer
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device
opt = getattr(config, 'Gis_Config')()

tokenizer = BertTokenizer.from_pretrained('./bert_pretrain/vocab/vocab_for_user_reputation')
def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def token_encode(str_list):
        '''
        1.Tokenize the sequence
        2.Encode the sequence
        '''
        str_list = tokenizer(str_list, padding='longest', return_tensors="pt")
        return str_list['input_ids'], str_list['token_type_ids'], str_list['attention_mask']


def collate_fn(batch):
    '''
    1. Tokenize the text here
    2. to Pytorch Tensor from list/ndaary
    '''
    label, q_id, a_id, q_title, q_body, q_tag, aid_q_titles, aid_q_bodys, aid_q_tags, bert_title = zip(*batch)

    b_str_list = tokenizer(bert_title, padding=True, truncation=True, max_length=512, return_tensors='pt')

    bid_input = b_str_list['input_ids']
    bid_type = b_str_list['token_type_ids']
    batt_mask = b_str_list['attention_mask']

    q_str_list = tokenizer(q_title, padding=True, truncation=True, max_length=512, return_tensors='pt')

    qid_input = q_str_list['input_ids']
    qid_type = q_str_list['token_type_ids']
    qatt_mask = q_str_list['attention_mask']


    label = torch.tensor(label)
    q_id = torch.tensor(q_id)
    a_id = torch.tensor(a_id)

    return label, q_id, a_id, bid_input, bid_type, batt_mask, qid_input, qid_type, qatt_mask


def run(**kwargs):

    global opt
    if 'dataset' in kwargs:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    datasetname = opt.dataset

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    model = getattr(models, opt.model)(opt)

    train_data = CQAData(opt.data_path, mode="Train")
    test_data = CQAData(opt.data_path, mode="Test")
    dev_data = CQAData(opt.data_path, mode="Dev")

    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    dev_data_loader = DataLoader(dev_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    accelerator.print(f'train data: {len(train_data)}; test data: {len(test_data)}; dev data: {len(dev_data)}')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    model, optimizer, train_data_loader = accelerator.prepare(model, optimizer, train_data_loader)
    test_data_loader = accelerator.prepare(test_data_loader)
    dev_data_loader = accelerator.prepare(dev_data_loader)


    bce = nn.BCEWithLogitsLoss()
    min_p1 = 1e-10

    accelerator.print("start training....")
    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        model.train()
        for idx, train_datas in enumerate(train_data_loader):
            label, _, data = train_datas[0], train_datas[1], train_datas[2:]
            optimizer.zero_grad()
            out = model(data)
            loss = bce(out.squeeze(1), label)
            total_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()

        scheduler.step()
        mean_loss = total_loss * 1.0 / idx
        accelerator.print(f"{now()}  Epoch {epoch}: train data: loss:{mean_loss:.4f}.")

        accelerator.print("dev data results")
        predict_p1 = dev(model, dev_data_loader, opt.metrics)

        if predict_p1 > min_p1:

            min_p1 = predict_p1
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            filename = 'checkpoints/' + 'Bert' + '_' + datasetname + '_' + 'best' + '.opt'
            accelerator.save(unwrapped_model.state_dict(), filename)

    time.sleep(120)
    accelerator.print("*****"*20)
    accelerator.print("test data results")
    filename = 'checkpoints/' + 'Bert' + '_' + datasetname + '_' + 'best' + '.opt'
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(torch.load(filename))
    test(unwrapped_model, test_data_loader, opt.metrics)
    accelerator.print("*****"*20)

def test(model, data_loader, metrics):
    model.eval()

    all_labels = []
    all_preds = []
    all_qid = []
    with torch.no_grad():
        for idx, test_data in enumerate(data_loader):
            label, q_id, data = test_data[0], test_data[1], test_data[2:]
            out = model(data)
            a_out = accelerator.gather(out)
            a_label = accelerator.gather(label)
            aq_id = accelerator.gather(q_id)
            all_preds.extend(np.reshape(a_out.cpu().numpy(), -1))
            all_labels.extend(a_label.cpu().numpy())
            all_qid.extend(aq_id.cpu().numpy())

    all_labels, all_preds = group_resuts(all_labels, all_preds, all_qid)

    res = cal_metric(all_labels, all_preds, metrics)
    res = [f"{k}: {v:.4f};" for k, v in res.items()]
    accelerator.print(' '.join(res))

def dev(model, data_loader, metrics):
    model.eval()
    all_labels = []
    all_preds = []
    all_qid = []
    accelerator.print('--------------------------------------')
    accelerator.print(now())
    with torch.no_grad():
        for idx, dev_data in enumerate(data_loader):
            label, q_id, data = dev_data[0], dev_data[1], dev_data[2:]
            out = model(data)
            a_out = accelerator.gather(out)
            a_label = accelerator.gather(label)
            aq_id = accelerator.gather(q_id)
            all_preds.extend(np.reshape(a_out.cpu().numpy(), -1))
            all_labels.extend(a_label.cpu().numpy())
            all_qid.extend(aq_id.cpu().numpy())
    all_labels, all_preds = group_resuts(all_labels, all_preds, all_qid)
    res = cal_metric(all_labels, all_preds, metrics)
    predict_p1 = res['P@1']
    res = [f"{k}: {v:.4f};" for k, v in res.items()]
    accelerator.print(' '.join(res))
    accelerator.print(now())
    return predict_p1


if __name__ == "__main__":
    fire.Fire()

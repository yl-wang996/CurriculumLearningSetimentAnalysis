# -*- coding: utf-8 -*-
# @Time    : 2021/11/27
# @Author  : Yuqian Lei & Yunlong Wang
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class Result():
    def __init__(self,statistics_path):
        self.statistics_path = statistics_path

    def _read_statstic(self, file,step_budget=0):
        if '.pt' not in file:
            return None,None
        name = file.split('/')[-1]
        ordering = name[:-3].split('_')[0]
        pacing_f = name[:-3].split('_')[1]
        useful_loss = []
        useful_acc = []
        with open(file, 'rb') as f:
            data = torch.load(f)
            assert data is not None
            test_loss = [n.item() for n in data['test_loss']]
            test_acc = [n.item() for n in data['test_acc']]
            iters = data['iter'][1:]
            if step_budget !=0:
                for idx,step in enumerate(iters):
                    if step <= step_budget:
                        useful_loss.append(test_loss[idx])
                        useful_acc.append(test_acc[idx])
            else:
                useful_loss = test_loss
                useful_acc = test_acc
        best_acc = max(useful_acc)
        best_loss = max(useful_loss)
        best_epoch = useful_acc.index(best_acc)
        if ordering  == 'standard':
            pacing_f = 'none'
        statistic = {
            'ordering': ordering,
            'pacing_f': pacing_f,
            'best_test_acc':best_acc,
            'best_test_loss':best_loss,
        }
        return statistic, best_epoch

    def get_statics(self):
        # statistics_N0.4
        best_epoch_sum = 0
        num = 0
        folders = os.listdir(self.statistics_path)
        statistic_all = []
        for folder in tqdm(folders):
            sta_path = os.path.join(self.statistics_path, folder)
            statistic,best_epoch = self._read_statstic(file=sta_path, step_budget=3015)
            if statistic is not None:
                statistic_all.append(statistic)
                best_epoch_sum += best_epoch
                num += 1
        data = {'data': statistic_all}
        return data

    def save2file(self,result_path):
        X_acc = []
        X_loss = []
        Y = []
        mapping = {
            'curr':0,
            'anti-curr':10,
            'random':20,
            'standard':30,
            'none':0,
            'exp':1,
            'linear':2,
            'log':3,
            'quad':4,
            'root':5,
            'step':6
        }
        markers = [ 'o','>','<','s','+','*','d']
        acc = {
            'curr':[],
            'anti-curr':[],
            'random':[],
            'standard':[]
        }
        best = {
            'curr':0,
            'anti-curr':0,
            'random':0,
            'standard':0
        }
        datas = self.get_statics()['data']
        for data in datas:
            y = 0
            X_acc.append(data['best_test_acc'])
            X_loss.append(data['best_test_loss'])
            y += mapping[data['ordering']]
            y += mapping[data['pacing_f']]
            Y.append(y)
            acc[data['ordering']].append(data['best_test_acc'])
            if data['best_test_acc']>best[data['ordering']]:
                best[data['ordering']] = data['best_test_acc']

        print('plotting')
        plt.figure()
        for idx, y in enumerate(Y):
            if y<10:
                color = 'darkcyan'
            elif y>=10 and y<20:
                color='purple'
            elif y>= 20 and y<30:
                color = 'yellowgreen'
            elif y>=30:
                color = 'black'
            a = y%10
            marker = markers[a]
            plt.scatter(X_acc[idx],y, c=color, marker=marker, alpha=0.4)

        plt.yticks([2.5,12.5,22.5,30],['Curr','AntiCurr','Random','standard'])
        plt.xlabel('IMDB Test Acc')
        plt.savefig(result_path)
        for order in list(acc.keys()):
            print(f'[{order}]: best_test_acc:{best[order]:.4f}, avg_test_acc:{np.average(acc[order]):.4f}, std_test_acc:{np.std(acc[order]):.4f}')

if __name__ == '__main__':
    pdf_name = 'result_N04.pdf'
    statistics_path = '/export/home/1wang/projects/CLinSematicClassification/statistics_N0.4/'
    result = Result(statistics_path=statistics_path)
    result.save2file(result_path=pdf_name)

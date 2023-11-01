# -*- coding: utf-8 -*-
from utils.data import MyDataset
import torch
from utils.utils import set_seed
import os
import numpy as np
import time
from utils.trainner import TrainerCL
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Config():
    '''
     a ∈ {0.1, 0.4, 0.8, 1.6} and b ∈ {0.1, 0.4, 0.8} over six pacing function families  =  72  results
    '''
    valp = 0.2  # percentage of validation dataset
    rand_fraction = 0.0  # to generate noisy data by fusion of labels
    ordering = 'curr'  # curr  /  anti-curr  / random  /  standard
    pacing_f = 'none'   # none / linear / quad / root / step / exp / log
    pacing_a = 1
    pacing_b = 1
    # for training
    epochs = 20
    batch_size = 10
    optimizer_name = 'sgd'
    lr = 0.01  # 0.1
    momentum = 0.9
    weight_decay = 1e-4
    scheduler_name = 'cosine'
    # other
    workers = 4
    seed = 2021
    printfreq = 20

    project_root = '/export/home/1wang/projects/CLinSematicClassification'
    data_root = '/data/1wang/data/CLinSematicClassification'
    statistic_path = os.path.join(project_root, 'statistics')
    statistic_file = None
    order_path = os.path.join(data_root, 'Logdir4LossScore/IMDB.order.pth')
    data_path = os.path.join(data_root, 'dataset/Subset_tr2500_test500_shuffled.npz')

    # choose CUDA device or CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main(config):
    print('-'*80)
    print(f'Time:{time.strftime("%Y-%m-%d %H:%M",time.localtime())}')
    if config.ordering != 'standard':
        print(f' start training with[odering: {config.ordering}  pacing_f:{config.pacing_f}  '
              f'pacing_a:{config.pacing_a}  pacing_b:{config.pacing_b}]')
    else:
        print(f'start training with[odering: {config.ordering}  lr:{config.lr}  optimizer:{config.optimizer_name}  '
              f'scheduler:{config.scheduler_name} batch_size:{config.batch_size}  weight_decay:{config.weight_decay} ]')
    print('-' * 80)
    set_seed(config.seed)

    # load dataset
    dataset = MyDataset(config.data_path)
    # test dataset always is clean
    train_dataset_clean, train_dataset, test_dataset = dataset.get_corrupt_dataset(config.rand_fraction)

    # Trying to get difficulty-order
    print(f'Trying to get difficuly-order')
    with open(config.order_path, 'rb') as f:
        dt = torch.load(f)
    order = list(dt.keys())

    # decide CL, Anti-CL, or random-CL
    if config.ordering == "random":
        np.random.shuffle(order)
    elif config.ordering == "anti-curr":
        order = [x for x in reversed(order)]
    elif config.ordering == "curr":
        pass
    else:
        assert config.ordering == "standard", 'please make sure, that order is right.'

    ordered_trainer = TrainerCL(
        config=config,
        order=order,
        train_dataset=train_dataset,
        train_dataset_clean = train_dataset_clean,
        test_dataset=test_dataset,
        statistic_path=config.statistic_file
    )
    ordered_trainer.run(record=True)


# get all the hyperparametes-setups to going
def get_todo_setups(statistic_path):
    def get_history(statistic_path):
        folders = os.listdir(statistic_path)
        setups = []
        for f in folders:
            if f[-3:] == '.pt':
                splits = f[:-3].split('_')
                setups.append(splits)
        return setups
    def generate_all_setups():
        # add all hyper-parameters as setup list for CL
        all_setups = []
        orderings = ['curr', 'anti-curr', 'random']  # curr  /  anti-curr  / random  /  standard
        # a ∈ {0.1, 0.4, 0.8, 1.6} and b ∈ {0.1, 0.4, 0.8} over six pacing function families  =  72  results
        pacing_f = ['linear', 'step', 'exp', 'log']  # ['linear', 'quad', 'root', 'step', 'exp', 'log','none']
        pacing_a = [0.1, 0.4, 0.8]  # [0.1, 0.4, 0.8, 1.6]
        pacing_b = [0.1, 0.4, 0.8]
        for ordering in orderings:
            for f in pacing_f:
                if f == 'none':
                    all_setups.append([ordering, f])
                    continue
                for a in pacing_a:
                    for b in pacing_b:
                        all_setups.append([ordering, f, str(a), str(b)])
        # add all hyper-parameters as setup list for standard training
        ordering = 'standard'
        lrs = [0.01, 0.001]
        optimizers = ['sgd', 'adam']
        schedulers = ['cosine', 'exponential']
        batchsizes = [5, 10]
        weightdecays = [0.0001, 0.00001, 0.000001, 0]
        for lr in lrs:
            for opt in optimizers:
                for scheduler in schedulers:
                    for bs in batchsizes:
                        for wd in weightdecays:
                            all_setups.append([ordering, str(lr), str(opt), str(scheduler), str(bs), str(wd)])
        return all_setups

    existed_setups = get_history(statistic_path)
    all_setups = generate_all_setups()
    todo_list = []
    for setup in all_setups:
        if setup not in existed_setups:
            todo_list.append(setup)
    return todo_list

# parse setups to config instance used for training
def parse_setup2config(setup:list,statistic_path,rand_fraction):
    config = Config()
    config.rand_fraction = rand_fraction
    config.statistic_path = statistic_path
    if rand_fraction !=0:
        config.order_path = os.path.join(config.data_root, f'Logdir4LossScore/IMDB_N{rand_fraction}.order.pth')
        config.data_path = os.path.join(config.data_root, f'dataset/Subset_tr2500_test500_shuffled_N{rand_fraction}.npz')
    if setup[0] == 'standard':
        config.ordering = str(setup[0])
        config.lr = float(setup[1])
        config.optimizer_name = str(setup[2])
        config.scheduler_name = str(setup[3])
        config.batch_size = int(setup[4])
        config.weight_decay = float(setup[5])
        config.statistic_file = os.path.join(config.statistic_path,
                                             f'{config.ordering}_{config.lr}_{config.optimizer_name}'
                                             f'_{config.scheduler_name}_{config.batch_size}_{str(config.weight_decay)}.pt')
    else:
        assert setup[0] in ['curr','anti-curr','random']
        config.ordering = str(setup[0])
        config.pacing_f = str(setup[1])
        if config.pacing_f != 'none':
            config.pacing_a = float(setup[2])
            config.pacing_b = float(setup[3])
            config.statistic_file = os.path.join(config.statistic_path,
                                                 f'{config.ordering}_{config.pacing_f}_{config.pacing_a}_'
                                                 f'{config.pacing_b}.pt')
        else:
            config.statistic_file = os.path.join(config.statistic_path,
                                                 f'{config.ordering}_{config.pacing_f}.pt')
    return config


def run():
    project_root = '/export/home/1wang/projects/CLinSematicClassification'
    statistic_path_wo_N = os.path.join(project_root, 'statistics')

    for rand_fraction in [0, 0.4]:
        if rand_fraction != 0:
            statistic_path = statistic_path_wo_N + f'_N{rand_fraction}'
        else:
            statistic_path = statistic_path_wo_N
        if not os.path.exists(statistic_path):
            os.mkdir(statistic_path)
        todo_list = get_todo_setups(statistic_path)
        for idx, setup in enumerate(todo_list):
            start_time = time.time()
            config = parse_setup2config(setup=setup,
                                   statistic_path=statistic_path,
                                   rand_fraction=rand_fraction)
            print(f'Runing START:   rand_fraction[{rand_fraction}]:  Setup:[{setup}  TODO[{len(todo_list)-idx-1}]   ]')
            main(config=config)
            print(f'Running END:    uesed time:  {(time.time()-start_time)/60:.2f} mins for each running')

if __name__ == '__main__':
    run()
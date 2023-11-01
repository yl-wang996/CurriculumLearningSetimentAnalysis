from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
import torchmetrics
from transformers import DistilBertConfig
from utils.utils import get_optimizer, LossTracker, get_pacing_function,\
    get_scheduler,balance_order_val
from torch import nn
from utils.dataset import Subset
import torch

class TrainerCL():
    def __init__(self,config,train_dataset,train_dataset_clean,test_dataset,order,statistic_path='stat_curr.pt'):
        # Spliting order to train set and validation set
        print(f'Spliting order to train set and validation set')
        self.order, self.order_val = balance_order_val(order, train_dataset, num_classes=2, valp=config.valp)
        self.train_dataset = train_dataset
        self.config = config
        self.statistic_path = statistic_path

        self.val_loader = torch.utils.data.DataLoader(Subset(train_dataset_clean, self.order_val), batch_size=self.config.batch_size,
                                                 shuffle=False, num_workers=self.config.workers, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.batch_size,
                                                  shuffle=False, num_workers=self.config.workers, pin_memory=True)
        self.model = self.build_model()
        self.optimizer = get_optimizer(
            optimizer_name=config.optimizer_name,
            parameters=self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
        self.scheduler = get_scheduler(
            scheduler_name=config.scheduler_name,
            optimizer=self.optimizer,
            num_epochs=config.epochs)

        self.loss_fn = nn.CrossEntropyLoss(reduction="none").cuda()
        self.acc_metric = torchmetrics.Accuracy()
        self.history ={"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
                       "test_loss": [], "test_acc": [],"iter": [0, ]}

    def build_model(self,pretraining=True):
        # init model
        if pretraining:
            # load the model with pretaining.
            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        else:
            # load the model without pretaining.
            distilbert_config = DistilBertConfig()
            model = DistilBertForSequenceClassification(distilbert_config)
        model.to(self.config.device)
        return model


    def validate(self,data_loader,prefix='val'):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            tracker = LossTracker(len(data_loader), prefix, self.config.printfreq)
            for i, batch in enumerate(data_loader):
                sentences = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                outputs = self.model(sentences, attention_mask=attention_mask, labels=labels)
                logits = outputs['logits']
                loss = outputs['loss']

                labels = labels.to('cpu')
                confidence = logits.softmax(dim=-1).to('cpu')
                acc = self.acc_metric(confidence, labels)

                tracker.update(loss, acc, self.config.batch_size)
                tracker.display(i)
        return tracker.losses.avg, tracker.acc.avg

    def run(self,record=True):
        bs = self.config.batch_size
        N = len(self.order)
        myiterations = (N // bs + 1) * self.config.epochs
        pre_iterations = 0
        startIter = 0


        pacing_function = get_pacing_function(myiterations, N, self.config)
        startIter_next = pacing_function(0)

        print('0 iter data between %s and %s with Pacing %s' % (startIter, startIter_next, self.config.pacing_f,))
        trainset = Subset(self.train_dataset, list(self.order[startIter:max(startIter_next, self.config.batch_size)]))  # train set should bigger than batch_size
        if self.config.ordering == 'standard':
            shuffle = True
        else:
            shuffle = False
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.config.batch_size,
                                                   shuffle=shuffle, num_workers=self.config.workers, pin_memory=True)
        epoch = 0
        step = 0
        while step < myiterations:
            epoch += 1
            tracker = LossTracker(len(train_loader), f'iteration : [{step}]', self.config.printfreq)
            self.model.train()
            for batch in train_loader:
                step += 1
                sentences = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                outputs = self.model(sentences, attention_mask=attention_mask, labels=labels)
                logits = outputs['logits']
                loss = outputs['loss']

                labels = labels.to('cpu')
                confidence = logits.softmax(dim=-1).to('cpu')
                acc = self.acc_metric(confidence, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                tracker.update(loss, acc, self.config.batch_size)
                tracker.display(step - pre_iterations)
            ## After each epoch
            # If we hit the end of the dynamic epoch build a new data loader
            pre_iterations = step
            if startIter_next <= N:
                startIter_next = pacing_function(step)  # <=======================================
                print("%s iter data between %s and %s w/ Pacing %s and LEARNING RATE %s " % (
                step, startIter, startIter_next, self.config.pacing_f, self.optimizer.param_groups[0]["lr"]))
                train_loader = torch.utils.data.DataLoader(
                    Subset(self.train_dataset, list(self.order[startIter:max(startIter_next, self.config.batch_size)])),
                    batch_size=self.config.batch_size,
                    shuffle=shuffle, num_workers=self.config.workers, pin_memory=True)

            # start your record
            if step > 50:
                tr_loss, tr_acc = tracker.losses.avg, tracker.acc.avg
                val_loss, val_acc = self.validate(self.val_loader,prefix='val')
                test_loss, test_acc = self.validate(self.test_loader,prefix='test')
                if record:
                    # record
                    self.history["test_loss"].append(test_loss)
                    self.history["test_acc"].append(test_acc)
                    self.history["val_loss"].append(val_loss)
                    self.history["val_acc"].append(val_acc)
                    self.history["train_loss"].append(tr_loss)
                    self.history["train_acc"].append(tr_acc)
                    self.history['iter'].append(step)
                    print(f'record as step {step}')
        torch.save(self.history, self.statistic_path)
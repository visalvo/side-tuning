from __future__ import division, print_function

import random
import time
from warnings import filterwarnings

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from datasets.sosnippets import SOSnippetsDataset
from models.nets import FusionSideNetFcResNet, FusionSideNetFcMobileNet, FusionSideNetDirect, FusionSideNetFcVGG
from models.utils import TrainingPipeline


filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

'''
alpha_configurations = [
    [0.2, 0.3, 0.5],
    [0.2, 0.4, 0.4],
    [0.2, 0.5, 0.3],
    [0.3, 0.2, 0.5],
    [0.3, 0.3, 0.4],
    [0.3, 0.4, 0.3],
    [0.3, 0.5, 0.2],
    [0.4, 0.2, 0.4],
    [0.4, 0.3, 0.3],
    [0.4, 0.4, 0.2],
    [0.5, 0.2, 0.3],
    [0.5, 0.3, 0.2]
]
'''

alpha_configurations = [
    [0.6, 0.4],
    [0.5, 0.5],
    [0.6, 0.4]
]


tes_dir = '/content/test'

if not os.path.exists(tes_dir):
        os.mkdir(tes_dir)

d = SOSnippetsDataset('/content/images', '/content/snippets')

d_train, d_val, d_test = torch.utils.data.random_split(d, [16000, 8000, 10000])
dl_train = DataLoader(d_train, batch_size=32, shuffle=True)
dl_val = DataLoader(d_val, batch_size=16, shuffle=True)
dl_test = DataLoader(d_test, batch_size=64, shuffle=False)

num_classes = len(d.classes)
train_targets = d_train.dataset.targets
labels = d.classes

num_epochs = 5

for alphas in alpha_configurations:
    for model in (
        # FusionSideNetFcMobileNet(300, num_classes=num_classes, alphas=alphas, dropout_prob=.5, side_fc=1024),
        # FusionSideNetFcMobileNet(300, num_classes=num_classes, alphas=alphas, dropout_prob=.5,
        #                         custom_embedding=True, custom_num_embeddings=len(w2v_model.wv.vocab), side_fc=512),
        FusionSideNetFcMobileNet(300, num_classes=num_classes, alphas=alphas, dropout_prob=.5,
                                 custom_embedding=True, custom_num_embeddings=len(w2v_model.wv.vocab), side_fc=1024),
        # FusionSideNetFcResNet(300, num_classes=num_classes, alphas=alphas, dropout_prob=.5, side_fc=1024),
        # FusionSideNetFcVGG(300, num_classes=num_classes, alphas=alphas, dropout_prob=.5, side_fc=1024),
    ):
        learning_rate = .1
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: learning_rate * (1.0 - float(epoch) / num_epochs) ** .5
        )
        pipeline = TrainingPipeline(model,
                                    criterion,
                                    optimizer,
                                    scheduler,
                                    num_classes=num_classes,
                                    best_model_path=f'{tes_dir}/models/best_{model.name}_model_{"-".join([str(i) for i in alphas])}.ptr')

        since = time.time()
        best_valid_acc, test_acc, cm, dist = pipeline.run(dl_train,
                                                          dl_val,
                                                          dl_test,
                                                          num_epochs=num_epochs,
                                                          classes=labels)
        time_elapsed = time.time() - since

        result_file = '/content/test/test-8.csv'
        with open(result_file, 'a+') as f:
            f.write(f'{model.name},'
                    f'{round(time_elapsed)},'
                    f'{sum(p.numel() for p in model.parameters() if p.requires_grad)},'
                    f'sgd,'
                    f'w2v,'
                    f'no,'
                    f'{"-".join([str(i) for i in alphas])},'
                    f'{best_valid_acc:.3f},'
                    f'{test_acc:.3f},'
                    f'{",".join([f"{r[i] / np.sum(r):.3f}" for i, r in enumerate(cm)])}\n')

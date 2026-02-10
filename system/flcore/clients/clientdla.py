import torch
import torch.nn as nn
import numpy as np
import copy
import sys
from flcore.clients.clientbase import Client
import torch.nn.functional as F

class clientDLAFed(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.base_layer_count = int(self.model.layer_count * self.args.base_layers)
        self.prev_predictor = None

    def train(self, adapt=False):
        trainloader = self.load_train_data()
        print("Loaded Data!")
        self.model.train()
        initial_layers = [copy.deepcopy(layer) for layer in self.model.layer_list]

        # Determine max local steps
        if adapt:
            self.model.unfreeze_layers()
            self.model.freeze_base()
            max_local_steps = self.args.plocal_steps
        else:
            self.model.unfreeze_layers()
            max_local_steps = self.local_steps

        xrayData = ["chexpert", "nihchestxray"]

        # Training loop
        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                # Move data to device
                if self.args.dataset in xrayData:
                    x = torch.stack(self.load_images(x)).to(self.device)
                elif isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)

                mu = getattr(self.args, 'mu', 0.01)
                prox_term = 0.0
                total_layers = len(self.model.layer_list)

                for i, (curr_layer, init_layer) in enumerate(zip(self.model.layer_list, initial_layers)):
                    weight = (total_layers - i) / total_layers
                    for p_curr, p_init in zip(curr_layer.parameters(), init_layer.parameters()):
                        if p_curr.requires_grad:
                            p_init = p_init.detach().to(p_curr.device)
                            prox_term += F.mse_loss(p_curr, p_init, reduction='sum') * weight

                loss += (mu / 2) * prox_term

                loss.backward()
                self.optimizer.step()



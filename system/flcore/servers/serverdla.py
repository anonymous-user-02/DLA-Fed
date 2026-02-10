import sys
import time
from flcore.clients.clientdla import clientDLAFed
from flcore.servers.serverbase import Server
import os
import logging
import torch
import statistics
import numpy as np
from torch import nn
import copy

class DLAFed(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, tmp:{args.temperature}, lam:{args.lambdaa}, model:{args.model_name}, num_clients:{args.num_clients}, base_layers:{args.base_layers}"
        clientObj = clientDLAFed
        self.message_hp_dash = self.message_hp.replace(", ", "-")
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}.h5")
        self.last_ckpt_fn = os.path.join(self.ckpt_dir, f"FedAvg-cifar10-100clt.pt")
        self.layer_count = args.model.layer_count
        self.layer_contribution_weights = [[1 for _ in range(args.num_clients)] for _ in range(self.layer_count)]

        self.set_clients(args, clientObj)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds):
            if i == 0:
                self.send_models()

            print(f"\n------------- Round number: [{i+1:3d}/{self.global_rounds}]-------------")
            print(f"==> Training for {len(self.clients)} clients...", flush=True)
            for client in self.clients:
                client.train(adapt=False)

            self.receive_models()
            if i == 0:
                self.aggregate_parameters()
            else:
                self.calculate_layer_contribution_weights()
                self.aggregate_parameters_layer_wise()

            self.send_models(mode="all")
            print("==> Evaluating models...", flush=True)
            mean_test_acc, best_mean_test_acc = self.evaluate()

            for client in self.clients:
                client.train(adapt=True)

            print("==> Evaluating Personalized models...", flush=True)
            # === Capture the returned accuracies ===
            mean_test_acc, best_mean_test_acc = self.evaluate()
            # === End of capture ===

            # === Add model saving and early stopping logic ===
            if mean_test_acc >= self.best_mean_test_acc:
                print(f"New best personalized accuracy: {mean_test_acc * 100:.2f}% (Round {i+1})")
                self.non_improve_rounds = 0

                for client in self.clients:
                    torch.save(client.model.state_dict(), f"DLAFed/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{self.args.base_layers}-{client.id}-{self.args.goal}-{self.times}.pth")
            elif i >= 9:
                self.non_improve_rounds += 1
                print(f"No improvement in personalized accuracy for {self.non_improve_rounds} consecutive round(s).", flush=True)
                if self.non_improve_rounds >= self.patience:
                    print("Early stopping triggered due to no improvement.", flush=True)
                    break

        print("==> Evaluating Personalized model Accuracy...", flush=True)
        for client in self.clients:
            client.model.load_state_dict(torch.load(f"DLAFed/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{self.args.base_layers}-{client.id}-{self.args.goal}-{self.times}.pth"))
        self.evaluate(val=False)

        self.save_results(fn=self.hist_result_fn)

    def standardize(self, x):
        mean = statistics.mean(x)
        std = statistics.stdev(x)
        if std != 0:
            for i in range(len(x)):
                x[i] = (x[i] - mean) / std
        return x

    def softmax(self, x):
        e_x = np.exp((x - np.max(x)) / self.args.temperature)
        softmax_probs = e_x / e_x.sum()
        return softmax_probs

    def zero_init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def calculate_layer_contribution_weights(self):
        client_layer_MSEs = [[] for _ in range(self.layer_count)]
        for layer_index in range(self.layer_count):
            averaged_layer_weights = self.global_model.layer_list[layer_index].weight.data
            for local_model in self.uploaded_models:
                local_layer_weights = local_model.layer_list[layer_index].weight.data
                MSE = -(torch.mean((averaged_layer_weights - local_layer_weights) ** 2).item())
                client_layer_MSEs[layer_index].append(MSE)

        layer_contribution_weights = []
        for i in range(len(client_layer_MSEs)):
            contribution_weights = self.args.lambdaa * self.softmax(self.standardize(client_layer_MSEs[i]))
            layer_contribution_weights.append(contribution_weights)
        self.layer_contribution_weights = layer_contribution_weights

    def aggregate_parameters_layer_wise(self):
        self.global_model.apply(self.zero_init_weights)
        for i in range(self.layer_count):
            weights = []
            biases = []

            for j in range(len(self.uploaded_models)):
                cont_weight = self.layer_contribution_weights[i][j]
                weights.append(self.uploaded_models[j].layer_list[i].weight.data * cont_weight)
                if self.global_model.layer_list[i].bias is not None:
                    biases.append(self.uploaded_models[j].layer_list[i].bias.data * cont_weight)
            print(self.layer_contribution_weights[i])

            if len(weights) > 0:
                stacked_weights = torch.stack(weights)
                self.global_model.layer_list[i].weight.data = torch.sum(stacked_weights, dim=0)
            if len(biases) > 0:
                stacked_biases = torch.stack(biases)
                self.global_model.layer_list[i].bias.data = torch.sum(stacked_biases, dim=0)


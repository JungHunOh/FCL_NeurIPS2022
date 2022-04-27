import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        # self.num_features = 512
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.fc.weight.data = self.fc.weight.data * 0 + 0.1

        if 'trans' in self.args.new_mode:
            tmp = [nn.Linear(self.num_features, self.num_features, bias=False), nn.ReLU(inplace=True), nn.Linear(self.num_features, 2, bias=False)]
            self.transformation_layer = nn.Sequential(*tmp)

            self.aux_fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        
        if 'learnfeat' in self.args.new_mode:
            self.gammas = nn.Parameter(torch.ones(self.args.num_classes-60), requires_grad=True)
            self.betas = nn.Parameter(torch.zeros(self.args.num_classes-60), requires_grad=True)


        self.t = 0

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            if 'trans' in self.args.new_mode:
                affine_param = self.transformation_layer(x)
                transformed_features = x * affine_param[:,0].unsqueeze(1) + affine_param[:,1].unsqueeze(1)
                transformed_features = F.linear(F.normalize(transformed_features, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
                transformed_features = self.args.temperature * transformed_features

                return transformed_features


                '''
                original = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.aux_fc.weight, p=2, dim=-1))
                original = original * self.args.temperature

                affine_param = self.transformation_layer(x)
                transformed_features = x * affine_param[:,0].unsqueeze(1) + affine_param[:,1].unsqueeze(1)
                transformed_features = F.linear(F.normalize(transformed_features, p=2, dim=-1), F.normalize(self.fc.weight[:60], p=2, dim=-1))
                transformed_features = self.args.temperature * transformed_features

                new = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight[60:], p=2, dim=-1))
                #new = F.linear(F.normalize(x * affine_param[:,0].unsqueeze(1) - affine_param[:,1].unsqueeze(1), p=2, dim=-1), F.normalize(self.fc.weight[60:], p=2, dim=-1))

                new = self.args.temperature * new

                x = torch.cat([transformed_features, new], dim=1)
                
                return x, original
                '''
            
            elif 'learnfeat' in self.args.new_mode:
                base = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight[:60], p=2, dim=-1)) * self.args.temperature
                new = []

                if self.training:
                    new = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight[60:], p=2, dim=-1)) * self.args.temperature
                else:
                    for i in range(60, 100):
                        new.append(F.linear(F.normalize(x * self.gammas[i-60] + self.betas[i-60], p=2, dim=-1), F.normalize(self.fc.weight[i].unsqueeze(0), p=2, dim=-1)) * self.args.temperature)
                    new = torch.cat(new, dim=1)

                return torch.cat([base,new], dim=1)
            else:
                x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
                x = self.args.temperature * x
                
        elif 'dot' in self.mode:
            x = self.fc(x)
        

        return x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')
    
    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()
            if 'trans' in self.args.new_mode:
                affine_params = self.transformation_layer(data)
                data = data * affine_params[:,0].unsqueeze(1) + affine_params[:,1].unsqueeze(1)

        if self.args.not_data_init and 'trans' not in self.args.new_mode:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

            if 'learnfeat' in self.args.new_mode:
                self.update_fc_avg(data, label, class_list,True)

    def update_fc_avg(self,data,label,class_list,after_ft=False):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            if after_ft:
                embedding = embedding * self.gammas[class_index-60] + self.betas[class_index-60]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc,session,gammas=None,betas=None):
        if 'learnfeat' in self.args.new_mode:
            base = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc[:60], p=2, dim=-1)) * self.args.temperature
            new = []
            for i in range(self.args.base_class, self.args.base_class + self.args.way * session):
                if i >= self.args.base_class + self.args.way * (session - 1) and i < self.args.base_class + self.args.way * session:
                    new.append(F.linear(F.normalize(x * gammas[i-(self.args.base_class + self.args.way * (session - 1))] + betas[i-(self.args.base_class + self.args.way * (session - 1))], p=2, dim=-1), F.normalize(fc[i].unsqueeze(0), p=2, dim=-1)) * self.args.temperature)
                else:
                    new.append(F.linear(F.normalize(x * self.gammas[i-60] + self.betas[i-60], p=2, dim=-1), F.normalize(fc[i].unsqueeze(0), p=2, dim=-1)) * self.args.temperature)
            new = torch.cat(new, dim=1)
            
            return torch.cat([base,new], dim=1)

        else:
            if 'dot' in self.args.new_mode:
                return F.linear(x,fc)
            elif 'cos' in self.args.new_mode:
                return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        if 'learnfeat' in self.args.new_mode:
            gammas = self.gammas[self.args.way * (session - 1):self.args.way * session].clone().detach()
            betas = self.betas[self.args.way * (session - 1):self.args.way * session].clone().detach()
            gammas.requires_grad = False
            betas.requires_grad = False
            optimized_parameters = [new_fc, gammas, betas]
            optimized_parameters = [{'params': optimized_parameters}]
        else:
            optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                if 'learnfeat' in self.args.new_mode:
                    logits = self.get_logits(data,fc,session,gammas,betas)
                else:
                    logits = self.get_logits(data,fc,session)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)
        if 'learnfeat' in self.args.new_mode:
            self.gammas.data[self.args.way * (session - 1):self.args.way * session].copy_(gammas.data)
            self.betas.data[self.args.way * (session - 1):self.args.way * session].copy_(betas.data)


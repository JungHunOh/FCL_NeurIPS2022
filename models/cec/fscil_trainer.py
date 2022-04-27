from models.base.fscil_trainer import FSCILTrainer as Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
import random

import matplotlib.pyplot as plt
import os

from .helper import *
from utils import *
from dataloader.data_utils import *
from .Network import MYNET


class FSCILTrainer(Trainer):
    def __init__(self, args):

        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.set_up_model()
        pass

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.base_mode)
        print(MYNET)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir != None:  #
            print('Loading init parameters from: %s' % self.args.model_dir)
            try:
                self.best_model_dict = torch.load(self.args.model_dir)['params']
            except:
                self.best_model_dict = torch.load(self.args.model_dir)

        else:
            print('*********WARNINGl: NO INIT MODEL**********')
            # raise ValueError('You must initialize a pre-trained model')
            pass

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def get_dataloader(self, session):
        if session == 0:
            if self.args.new_loader:
                trainset, trainloader, testloader, new_trainloader = self.get_base_dataloader_meta()

                return trainset, trainloader, testloader, new_trainloader
            else:
                trainset, trainloader, testloader = self.get_base_dataloader_meta()

                return trainset, trainloader, testloader
        else:
            trainset, trainloader, testloader = self.get_new_dataloader(session)
            
            return trainset, trainloader, testloader

    def get_base_dataloader_meta(self):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(0 + 1) + '.txt'
        class_index = np.arange(self.args.base_class)
        if self.args.dataset == 'cifar100':
            # class_index = np.arange(self.args.base_class)
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=True,
                                                  index=class_index, base_sess=True)
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_index, base_sess=True)

        if self.args.dataset == 'cub200':
            # class_index = np.arange(self.args.base_class)
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_index)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_index)

        # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

        if self.args.new_loader:
            sampler = CategoriesSampler(trainset.targets, self.args.train_episode, self.args.num_query_base_class, self.args.num_query_base)
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=8,
                                                      pin_memory=True)
            
            if self.args.fullproto:
                new_sampler = CategoriesSampler(trainset.targets, self.args.train_episode, 45, self.args.num_fullproto + 5 + self.args.num_query_new)
            else:
                new_sampler = CategoriesSampler(trainset.targets, self.args.train_episode, 5, 45)
            new_trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=new_sampler, num_workers=8,
                                                      pin_memory=True)
            return trainset, trainloader, testloader, new_trainloader
        else:
            sampler = CategoriesSampler(trainset.targets, self.args.train_episode, self.args.episode_way,
                                        self.args.episode_shot +  + self.args.episode_query)
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=8,
                                                      pin_memory=True)

            return trainset, trainloader, testloader

    def get_new_dataloader(self, session):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(session + 1) + '.txt'
        if self.args.dataset == 'cifar100':
            class_index = open(txt_path).read().splitlines()
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=False,
                                                  index=class_index, base_sess=False)
        if self.args.dataset == 'cub200':
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True,
                                                index_path=txt_path)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True,
                                                      index_path=txt_path)
        if self.args.batch_size_new == 0:
            batch_size_new = trainset.__len__()
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                      num_workers=8, pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.args.batch_size_new,
                                                      shuffle=True,
                                                      num_workers=8, pin_memory=True)

        class_new = self.get_session_classes(session)

        if self.args.dataset == 'cifar100':
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_new, base_sess=False)
        if self.args.dataset == 'cub200':
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_new)
        if self.args.dataset == 'mini_imagenet':
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_new)

        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.args.test_batch_size, shuffle=False,
                                                 num_workers=8, pin_memory=True)

        return trainset, trainloader, testloader

    def get_session_classes(self, session):
        class_list = np.arange(self.args.base_class + session * self.args.way)
        return class_list

    def replace_to_rotate(self, proto_tmp, query_tmp, pseudo_tmp, full_tmp=None):
        selected_rots = []
        for i in range(self.args.low_way):
            # random choose rotate degree
            rot_list = [90, 180, 270]
            sel_rot = random.choice(rot_list)
            selected_rots.append(sel_rot)
            if sel_rot == 90:  # rotate 90 degree
                # print('rotate 90 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
                if full_tmp is not None:
                    full_tmp[i::self.args.low_way] = full_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
            elif sel_rot == 180:  # rotate 180 degree
                # print('rotate 180 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].flip(2).flip(3)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].flip(2).flip(3)
                if full_tmp is not None:
                    full_tmp[i::self.args.low_way] = full_tmp[i::self.args.low_way].flip(2).flip(3)
            elif sel_rot == 270:  # rotate 270 degree
                # print('rotate 270 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
                if full_tmp is not None:
                    full_tmp[i::self.args.low_way] = full_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
        
        num_pseudo_new_classes = int(pseudo_tmp.shape[0] / (5 + self.args.num_query_new))

        for i in range(num_pseudo_new_classes):
            rot_list = [90, 180, 270]
            sel_rot = random.choice(rot_list)

            if sel_rot == 90:
                pseudo_tmp[i::num_pseudo_new_classes] = pseudo_tmp[i::num_pseudo_new_classes].transpose(2, 3).flip(2)
            elif sel_rot == 180:
                pseudo_tmp[i::num_pseudo_new_classes] = pseudo_tmp[i::num_pseudo_new_classes].flip(2).flip(3)
            elif sel_rot == 270:
                pseudo_tmp[i::num_pseudo_new_classes] = pseudo_tmp[i::num_pseudo_new_classes].transpose(2,3).flip(3)
        
        if full_tmp is not None:
            return proto_tmp, query_tmp, pseudo_tmp, full_tmp, selected_rots
        else:
            return proto_tmp, query_tmp, pseudo_tmp, selected_rots
    
    def mixup(self, proto_tmp, query_tmp, pseudo_tmp, full_tmp=None):
        #proto_tmp [25, 3, 32 ,32]
        #query_tmp [200, 3, 32 ,32]
        
        index = torch.randperm(5)
        
        p = proto_tmp.clone()
        q = query_tmp.clone()
        
        if full_tmp is not None:
            f = full_tmp.clone()

        lams = []

        for i in range(5):
            lam = random.uniform(0.1, 0.9)
            lams.append(lam)
            
            proto_tmp[i::self.args.low_way] = lam * proto_tmp[i::self.args.low_way] + (1-lam) * p[index[i]::self.args.low_way]
            query_tmp[i::self.args.low_way] = lam * query_tmp[i::self.args.low_way] + (1-lam) * q[index[i]::self.args.low_way]

            if full_tmp is not None:
                full_tmp[i::self.args.low_way] = lam * full_tmp[i::self.args.low_way] + (1-lam) * f[index[i]::self.args.low_way]
        
        pseudo = pseudo_tmp.clone()
        num_pseudo_new_classes = int(pseudo_tmp.shape[0] / (5 + self.args.num_query_new))
        index = torch.randperm(num_pseudo_new_classes)

        for i in range(num_pseudo_new_classes):
            lam = random.uniform(0.1, 0.9)

            pseudo_tmp[i::num_pseudo_new_classes] = lam * pseudo_tmp[i::num_pseudo_new_classes] + (1-lam) * pseudo[index[i]::num_pseudo_new_classes]
        
        return proto_tmp, query_tmp, pseudo_tmp, full_tmp


    def get_optimizer_base(self):
        if self.args.meta:
            if self.args.meta_sgd:
                optimizer = torch.optim.SGD([{'params': self.model.module.proto_estimator.parameters(), 'lr': self.args.outer_lr},
                                            {'params': self.model.module.meta_sgd_params.parameters(), 'lr': self.args.outer_lr}],
                                            #{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base}],
                                            momentum=0.9, nesterov=True, weight_decay=self.args.decay)
                #optimizer = torch.optim.Adam([{'params': self.model.module.proto_estimator.parameters(), 'lr': self.args.outer_lr},
                #                            {'params': self.model.module.meta_sgd_params.parameters(), 'lr': self.args.outer_lr}],
                #                            #{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base}],
                #                            )
            else:
                if self.args.feat_transform:
                    optimizer = torch.optim.SGD([{'params': self.model.module.proto_estimator.parameters(), 'lr': self.args.outer_lr},
                                                {'params': self.model.module.feat_transformer.parameters(), 'lr': self.args.outer_lr},
                                                {'params': self.model.module.fc.parameters(), 'lr': self.args.lr_base},
                                                {'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base}],
                                                momentum=0.9, nesterov=True, weight_decay=self.args.decay)
                else:
                    optimizer = torch.optim.SGD([{'params': self.model.module.proto_estimator.parameters(), 'lr': self.args.outer_lr},
                                                {'params': self.model.module.fc.parameters(), 'lr': self.args.lr_base},
                                                {'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base}],
                                                momentum=0.9, nesterov=True, weight_decay=self.args.decay)
                    #optimizer = torch.optim.Adam([{'params': self.model.module.proto_estimator.parameters(), 'lr': self.args.outer_lr}],
                    #                            #{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base}],
                    #                            )
        else:
            optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                         {'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lrg}],
                                        momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):
            
            if self.args.new_loader and session == 0:
                train_set, trainloader, testloader, new_trainloader = self.get_dataloader(session)
            else:
                train_set, trainloader, testloader = self.get_dataloader(session)

            self.model = self.update_param(self.model, self.best_model_dict)

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                #if self.args.meta:
                #    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                '''
                self.model.eval()
                for c in range(60):
                    data = []
                    for i in range(500):
                        tmp = train_set[i+500*c][0]
                        tmp1 = tmp.transpose(1,2).flip(2)
                        data.append(tmp1)
                    data = torch.stack(data)
                    encoded = self.model.module.encode(data.cuda())
                    proto = encoded.mean(dim=0)
                    
                    torch.save(proto, f'./full_data_proto/class{c}_rot270.pt')
                import pdb; pdb.set_trace()
                '''

                for epoch in range(args.epochs_base):
                    
                    start_time = time.time()
                    # train base sess
                    self.model.eval()
                    #self.model.train()
                    
                    print(self.args.save)

                    if epoch == self.args.freeze_epoch:
                        self.model.module.fc.weight.requires_grad = False
                        for p in self.model.module.encoder.parameters():
                            p.requires_grad = False
                        
                        optimizer.param_groups[1]['weight_decay'] = 0
                        optimizer.param_groups[2]['weight_decay'] = 0
                        optimizer.param_groups[1]['momentum'] = 0
                        optimizer.param_groups[2]['momentum'] = 0
                        
                        #self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)

                    #self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)

                    if self.args.new_loader:
                        tl, ta = self.base_train(self.model, trainloader, optimizer, scheduler, epoch, args, new_trainloader)
                    else:
                        tl, ta = self.base_train(self.model, trainloader, optimizer, scheduler, epoch, args)

                    #if epoch < self.args.epochs_base-1:
                    #self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)

                    self.model.module.mode = 'avg_cos'

                    if args.set_no_val or args.meta: # set no validation
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        tsl, tsa = self.test(self.model, testloader, args, session)
                        self.trlog['test_loss'].append(tsl)
                        self.trlog['test_acc'].append(tsa)
                        lrc = scheduler.get_last_lr()[0]
                        print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                        result_list.append(
                            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))
                    else:
                        # take the last session's testloader for validation
                        vl, va = self.validation()

                        # save better model
                        if (va * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                            torch.save(dict(params=self.model.state_dict()), save_model_dir)
                            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            print('********A better model is found!!**********')
                            print('Saving model to :%s' % save_model_dir)
                        print('best epoch {}, best val acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                          self.trlog['max_acc'][session]))
                        self.trlog['val_loss'].append(vl)
                        self.trlog['val_acc'].append(va)
                        lrc = scheduler.get_last_lr()[0]
                        print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                            epoch, lrc, tl, ta, vl, va))
                        result_list.append(
                            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                                epoch, lrc, tl, ta, vl, va))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)

                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                # always replace fc with avg mean
                self.model.load_state_dict(self.best_model_dict)
                #self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                torch.save(dict(params=self.model.state_dict()), best_model_dir)

                self.model.module.mode = 'avg_cos'
                tsl, tsa = self.test(self.model, testloader, args, session)
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                print('The test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                self.plot_cos_sim(session)


            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                self.model.load_state_dict(self.best_model_dict)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                if self.args.meta:
                    self.model.module.estimate_prototype(trainloader, np.unique(train_set.targets), session)
                else:
                    self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                tsl, tsa, tsa_base, tsa_new  = self.test(self.model, testloader, args, session)

                # save better model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                self.trlog['max_acc_base'][session-1] = float('%.3f' % (tsa_base * 100))
                self.trlog['max_acc_new'][session-1] = float('%.3f' % (tsa_new * 100))

                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))
                
                self.plot_cos_sim(session)

        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60

        result_list.append('Best epoch:%d' % self.trlog['max_acc_epoch'])
        result_list.append('Total Acc.\n{}'.format(self.trlog['max_acc']))
        result_list.append('Base Acc.\n{}'.format(self.trlog['max_acc_base']))
        result_list.append('New Acc.\n{}'.format(self.trlog['max_acc_new']))
        print('Best epoch:', self.trlog['max_acc_epoch'])
        print('Total Acc.\n',self.trlog['max_acc'])
        print('Base Acc.\n',self.trlog['max_acc_base'])
        print('New Acc.\n',self.trlog['max_acc_new'])
        print('Total time used %.2f mins' % total_time)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

    def validation(self):
        with torch.no_grad():
            model = self.model

            for session in range(1, self.args.sessions):
                train_set, trainloader, testloader = self.get_dataloader(session)

                trainloader.dataset.transform = testloader.dataset.transform
                model.module.mode = 'avg_cos'
                model.eval()
                model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                vl, va = self.test(model, testloader, self.args, session)

        return vl, va

    def base_train(self, model, trainloader, optimizer, scheduler, epoch, args, new_trainloader=None):
        tl = Averager()
        ta = Averager()

        tqdm_gen = tqdm(trainloader)

        if self.args.new_loader:
            tqdm_gen_new = tqdm(new_trainloader)

            support_label = torch.arange(args.episode_way).repeat(args.episode_shot) + args.base_class
            query_label = torch.arange(args.episode_way).repeat(self.args.num_query_new) + args.base_class
            support_label = support_label.type(torch.cuda.LongTensor)
            query_label = query_label.type(torch.cuda.LongTensor)

            for i, (batch, batch_new) in enumerate(zip(tqdm_gen, tqdm_gen_new), 1):
                data, true_label = [_.cuda() for _ in batch] # This is for query_base
                data_new, true_label_new = [_.cuda() for _ in batch_new] # This is for query_new, support and pseudo_new_class
                
                num_new_classes = 5
                num_pseudo_new_classes = random.randint(20,39)
                num_whole = 5 + self.args.num_query_new + self.args.num_fullproto
                index_new = np.tile(np.arange(num_new_classes), num_whole) + np.repeat(np.arange(num_whole)*45,num_new_classes)
                index_pseudo_new = np.tile(np.arange(num_new_classes,num_new_classes+num_pseudo_new_classes),5 + self.args.num_query_new) + np.repeat(np.arange(5 + self.args.num_query_new)*45,num_pseudo_new_classes)

                data_pseudo_new = data_new[index_pseudo_new]
                true_label_pseudo_new = true_label_new[index_pseudo_new]

                data_new = data_new[index_new]
                true_label_new = true_label_new[index_new]

                s_l = support_label + num_pseudo_new_classes
                q_l = torch.cat([true_label, query_label+num_pseudo_new_classes])

                if self.args.fullproto:
                    proto, query = data_new[:25], data_new[25:25+5*self.args.num_query_new]
                    fulldata = data_new[25+5*self.args.num_query_new:]

                    full_tmp = deepcopy(
                    fulldata.reshape(self.args.num_fullproto, 5, fulldata.shape[1], fulldata.shape[2], fulldata.shape[3])[:,
                    :args.low_way, :, :, :].flatten(0, 1))

                else:
                    proto, query = data_new[:25], data_new[25:]

                # sample low_way data
                proto_tmp = deepcopy(
                    proto.reshape(args.episode_shot, args.episode_way, proto.shape[1], proto.shape[2], proto.shape[3])[
                    :args.low_shot,
                    :args.low_way, :, :, :].flatten(0, 1))
                query_tmp = deepcopy(
                    query.reshape(args.num_query_new, args.episode_way, query.shape[1], query.shape[2], query.shape[3])[:,
                    :args.low_way, :, :, :].flatten(0, 1))

                pseudo_tmp = deepcopy(
                    data_pseudo_new.reshape(5 + self.args.num_query_new, num_pseudo_new_classes, data_pseudo_new.shape[1], data_pseudo_new.shape[2], data_pseudo_new.shape[3])[:,
                    :num_pseudo_new_classes, :, :, :].flatten(0, 1))

                # random choose rotate degree
                if self.args.fullproto:
                    proto_tmp, query_tmp, pseudo_tmp, full_tmp, selected_rots = self.replace_to_rotate(proto_tmp, query_tmp, pseudo_tmp, full_tmp) # proto [50, 3, 32, 32] query [200, 3, 32, 32]
                else:
                    proto_tmp, query_tmp, pseudo_tmp, selected_rots = self.replace_to_rotate(proto_tmp, query_tmp, pseudo_tmp) # proto [50, 3, 32, 32] query [200, 3, 32, 32]

                if self.args.mixup:
                    if self.args.fullproto:
                        proto_tmp, query_tmp, pseudo_tmp, full_tmp = self.mixup(proto_tmp, query_tmp, pseudo_tmp, full_tmp)
                    else:
                        proto_tmp, query_tmp, pseudo_tmp, _ = self.mixup(proto_tmp, query_tmp, pseudo_tmp)
                        
                #proto_tmp: support, query_tmp: query_new, data: query_base
                
                model.eval()
                model.module.mode = 'encoder'
                data = model(data) # [250, 64]
                #with torch.no_grad():
                #proto_tmp = model(proto_tmp) # [50, 64]
                #with torch.no_grad():
                query_tmp = model(query_tmp) # 
                proto_tmp = model(proto_tmp) # support

                if num_pseudo_new_classes > 0:
                    with torch.no_grad():
                        pseudo_tmp = model(pseudo_tmp)
                if self.args.fullproto:
                    with torch.no_grad():
                        full_tmp = model(full_tmp)
                
                if num_pseudo_new_classes > 0:
                    pseudo_proto = []
                    for jj in range(num_pseudo_new_classes):
                        pseudo_proto.append(pseudo_tmp[jj::num_pseudo_new_classes].mean(dim=0).clone())
                
                    pseudo_proto = torch.stack(pseudo_proto)
                    #pseudo_proto = F.normalize(pseudo_proto, p=2, dim=-1) + torch.normal(0,0.0005, pseudo_proto.shape).cuda()
                else:
                    pseudo_proto = None

                #model.train()
                support = [proto_tmp, proto_tmp]
                query = [data, query_tmp]

                #target_proto = []
                #for i,l in enumerate(true_label[:5]):
                #    tmp = torch.load(f'./full_data_proto/class{l}_rot{selected_rots[i]}.pt')
                #    target_proto.append(tmp)
                #for i in range(60):
                #    tmp = torch.load(f'./full_data_proto/class{i}.pt')
                #    target_proto.append(tmp)
                #target_proto = torch.stack(target_proto)
                #target_proto = torch.cat([self.model.module.fc.weight.data[:60], target_proto], dim=0)
                #logits = self.args.temperature * F.linear(F.normalize(torch.cat(query,dim=0), p=2, dim=-1), F.normalize(target_proto, p=2, dim=-1))
                #logits = self.args.temperature * F.linear(F.normalize(query[1], p=2, dim=-1), F.normalize(target_proto, p=2, dim=-1))
                if self.args.data_dropout:
                    logits = []
                    estimated_proto = []
                    for jj in range(3):
                        if jj == 0:
                            a, b = model.module._forward(support, query, s_l, data_drop=False)
                        else:
                            a, b = model.module._forward(support, query, s_l)
                        logits.append(a)
                        estimated_proto.append(b)
                    
                    loss = 0

                    for jj, (logit, e_p) in enumerate(zip(logits, estimated_proto)):
                        loss = loss + F.cross_entropy(logit, q_l)
                        #estimated_proto[jj] = F.normalize(e_p, p=2, dim=-1)
                    
                    dropcontrastive_loss = 0
                    #for jj in range(3):
                        #dropcontrastive_loss = dropcontrastive_loss + torch.dot(estimated_proto[0][jj], estimated_proto[1][jj]) + torch.dot(estimated_proto[0][jj], estimated_proto[2][jj]) + torch.dot(estimated_proto[1][jj], estimated_proto[2][jj])
                    dropcontrastive_loss = abs(estimated_proto[0]-estimated_proto[1]).mean() + abs(estimated_proto[0]-estimated_proto[2]).mean() + abs(estimated_proto[1]-estimated_proto[2]).mean()

                    target_proto = []
                    for i,l in enumerate(true_label_new[:5]):
                        tmp = torch.load(f'./full_data_proto/class{l}_rot{selected_rots[i]}.pt')
                        target_proto.append(tmp)
                    target_proto = torch.stack(target_proto)
                    
                    proto_loss = 0
                    for jj in range(3):
                        proto_loss = proto_loss + abs(target_proto - estimated_proto[jj]).mean()

                    total_loss = loss + dropcontrastive_loss * self.args.dropcontrastive_lamb + proto_loss * self.args.proto_lamb
                    acc = count_acc(logits[0], q_l)

                else:
                    logits, estimated_proto, pseudo_proto_list = model.module._forward(support, query, s_l, pseudo_proto)
                    estimated_proto = torch.stack(estimated_proto)

                    if self.args.fullproto:
                        target_proto = []
                        for jj in range(5):
                            target_proto.append(full_tmp[jj::5].mean(dim=0))
                        target_proto = torch.stack(target_proto)
                        
                        proto_loss = 0
                        target_proto = F.normalize(target_proto, p=2, dim=-1)
                        estimated_proto_tmp = F.normalize(estimated_proto, p=2, dim=-1)

                        for jj in range(target_proto.shape[0]):
                            proto_loss = proto_loss + torch.dot(target_proto[jj], estimated_proto_tmp[jj])
                        proto_loss = proto_loss / target_proto.shape[0]
                        #proto_loss = abs(target_proto - estimated_proto).mean()
                    else:
                        proto_loss = 0
                    
                    '''
                    #
                    target_proto = []
                    for i,l in enumerate(true_label_new[:5]):
                        tmp = torch.load(f'./full_data_proto/class{l}_rot{selected_rots[i]}.pt')
                        target_proto.append(tmp)
                    target_proto = torch.stack(target_proto)
                    
                    #proto_loss = 0
                    #target_proto = F.normalize(target_proto, p=2, dim=-1)
                    #estimated_proto = F.normalize(estimated_proto, p=2, dim=-1)

                    #for i in range(target_proto.shape[0]):
                    #    proto_loss = proto_loss + torch.dot(target_proto[i], estimated_proto[i])
                    proto_loss = abs(target_proto - estimated_proto).mean()
                    #import pdb; pdb.set_trace()
                    '''
                    #tt = torch.cat([self.model.module.fc.weight.data[:60], target_proto], dim=0)
                    #logits = self.args.temperature * F.linear(F.normalize(torch.cat(query,dim=0), p=2, dim=-1), F.normalize(tt, p=2, dim=-1))
                    #logits = self.args.temperature * F.linear(F.normalize(query[1], p=2, dim=-1), F.normalize(target_proto, p=2, dim=-1))
                    
                    loss_base = 0
                    loss_new = 0
                    contrastive_loss = 0
                    logits = []
                    for jj in range(5):
                        if pseudo_proto_list[jj] is not None:
                            num_cur_pseudo_new_classes = pseudo_proto_list[jj].shape[0]
                        else:
                            num_cur_pseudo_new_classes = 0

                        if num_cur_pseudo_new_classes > 0:
                            tmp = [self.model.module.fc.weight[:60], pseudo_proto_list[jj], estimated_proto[jj].unsqueeze(0)]
                        else:
                            tmp = [self.model.module.fc.weight[:60], estimated_proto[jj].unsqueeze(0)]
                        tmp = torch.cat(tmp, dim=0)

                        output = self.args.temperature * F.linear(F.normalize(query[0], p=2, dim=-1), F.normalize(tmp, p=2, dim=-1))
                        loss_base = loss_base + F.cross_entropy(output, true_label)
                        
                        if num_cur_pseudo_new_classes > 0:
                            tmp = [self.model.module.fc.weight[:60].detach(), pseudo_proto_list[jj], estimated_proto[jj].unsqueeze(0)]
                        else:
                            tmp = [self.model.module.fc.weight[:60].detach(), estimated_proto[jj].unsqueeze(0)]
                        tmp = torch.cat(tmp, dim=0)

                        output = self.args.temperature * F.linear(F.normalize(query[1][jj::5], p=2, dim=-1), F.normalize(tmp, p=2, dim=-1))

                        assert output.shape[-1] == 60 + num_cur_pseudo_new_classes + 1
                        
                        loss_new = loss_new + F.cross_entropy(output, query_label[0::5] + num_cur_pseudo_new_classes)
                        logits.append(self.args.temperature * F.linear(F.normalize(query[1][jj::5], p=2, dim=-1), F.normalize(tmp.detach(), p=2, dim=-1)))
                        #loss = loss + F.cross_entropy(output, torch.cat([true_label, query_label[0::5]]))
                        contrastive_loss = 0#contrastive_loss + F.linear(estimated_proto[jj].unsqueeze(0), F.normalize(tmp[:60+num_cur_pseudo_new_classes].detach(), p=2, dim=-1)).mean()

                    fc_loss = F.linear(F.normalize(self.model.module.fc.weight[:60],p=2,dim=-1), F.normalize(self.model.module.fc.weight[:60],p=2,dim=-1))
                    #print('1',fc_loss.min().item(), fc_loss.mean().item())
                    fc_loss = fc_loss.mean()
                    #print(f'##### {loss_base.item() / 5}, {loss_new.item()/5} #########')
                    total_loss = (loss_base + loss_new * self.args.query_new_lamb) / (5+5*self.args.query_new_lamb) - proto_loss * self.args.proto_lamb + contrastive_loss / 5 * self.args.lamb_contrastive + fc_loss * self.args.fc_lamb
                    #total_loss = F.cross_entropy(logits, q_l) + proto_loss * self.args.proto_lamb + transform_loss * self.args.trans_lamb

                    #if num_pseudo_new_classes > 0:
                    #    tmp = [self.model.module.fc.weight.data[:60], pseudo_proto, estimated_proto]
                    #else:
                    #    tmp = [self.model.module.fc.weight.data[:60], estimated_proto]
                    #tmp = torch.cat(tmp, dim=0)
                    #logits = self.args.temperature * F.linear(F.normalize(torch.cat(query, dim=0), p=2, dim=-1), F.normalize(tmp.detach(), p=2, dim=-1))
                    acc = 0
                    for jj in range(5):
                        #tmp = count_acc(logits[jj], torch.cat([true_label, query_label[0::5]]))
                        acc += count_acc(logits[jj], query_label[0::5] - 60 + logits[jj].shape[-1] - 1)
                        #acc += count_acc(logits[jj], true_label)
                    acc = acc / 5
                    #acc = count_acc(logits, q_l)
    
                lrc = scheduler.get_last_lr()[0]
                tqdm_gen.set_description(
                    'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
                tl.add(total_loss.item())
                ta.add(acc)

                #print(self.model.module.fc.weight.sum())
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        else:
            if not args.meta:
                label = torch.arange(args.episode_way + args.low_way).repeat(args.episode_query)
                label = label.type(torch.cuda.LongTensor)
            else:
                support_label = torch.arange(args.episode_way).repeat(args.episode_shot) + args.base_class
                query_label = torch.arange(args.episode_way).repeat(args.episode_query) + args.base_class

                support_label = support_label.type(torch.cuda.LongTensor)
                query_label = query_label.type(torch.cuda.LongTensor)

            for i, batch in enumerate(tqdm_gen, 1):
                data, true_label = [_.cuda() for _ in batch] # [250, 3, 32, 32]


                k = args.episode_way * args.episode_shot
                if args.meta:
                    #s_l = torch.cat([true_label[:k], support_label])
                    s_l = support_label
                    q_l = torch.cat([true_label[k:], query_label])

                proto, query = data[:k], data[k:]
                # sample low_way data
                proto_tmp = deepcopy(
                    proto.reshape(args.episode_shot, args.episode_way, proto.shape[1], proto.shape[2], proto.shape[3])[
                    :args.low_shot,
                    :args.low_way, :, :, :].flatten(0, 1))
                query_tmp = deepcopy(
                    query.reshape(args.episode_query, args.episode_way, query.shape[1], query.shape[2], query.shape[3])[:,
                    :args.low_way, :, :, :].flatten(0, 1))

                if self.args.mixup:
                    proto_tmp, query_tmp = self.mixup(proto_tmp, query_tmp) # proto [50, 3, 32, 32] query [200, 3, 32, 32]

                # random choose rotate degree
                proto_tmp, query_tmp, selected_rots = self.replace_to_rotate(proto_tmp, query_tmp) # proto [50, 3, 32, 32] query [200, 3, 32, 32]

                model.module.mode = 'encoder'
                data = model(data) # [250, 64]
                proto_tmp = model(proto_tmp) # [50, 64]
                query_tmp = model(query_tmp) # [200, 64]

                if not args.meta:
                    # k = args.episode_way * args.episode_shot
                    proto, query = data[:k], data[k:]

                    proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1]) # [5, 10, 64] # [shot, way, feat]
                    query = query.view(args.episode_query, args.episode_way, query.shape[-1]) # [20, 10, 64]

                    proto_tmp = proto_tmp.view(args.low_shot, args.low_way, proto.shape[-1]) # [5, 10, 64]
                    query_tmp = query_tmp.view(args.episode_query, args.low_way, query.shape[-1]) # [20, 10, 64]

                    proto = proto.mean(0).unsqueeze(0)
                    proto_tmp = proto_tmp.mean(0).unsqueeze(0)

                    proto = torch.cat([proto, proto_tmp], dim=1) # [1, 20, 64]
                    query = torch.cat([query, query_tmp], dim=1) # [20, 20, 64]

                    proto = proto.unsqueeze(0)
                    query = query.unsqueeze(0)

                    logits = model.module._forward(proto, query)

                    total_loss = F.cross_entropy(logits, label)
                    acc = count_acc(logits, label)
                else:
                    support = [data[:k], proto_tmp]
                    query = [data[k:], query_tmp]

                    target_proto = []
                    #for i,l in enumerate(true_label[:5]):
                    #    tmp = torch.load(f'./full_data_proto/class{l}_rot{selected_rots[i]}.pt')
                    #    target_proto.append(tmp)
                    #for i in range(60):
                    #    tmp = torch.load(f'./full_data_proto/class{i}.pt')
                    #    target_proto.append(tmp)
                    #target_proto = torch.stack(target_proto)
                    #target_proto = torch.cat([self.model.module.fc.weight.data[:60], target_proto], dim=0)
                    #logits = self.args.temperature * F.linear(F.normalize(torch.cat(query,dim=0), p=2, dim=-1), F.normalize(target_proto, p=2, dim=-1))
                    #logits = self.args.temperature * F.linear(F.normalize(query[1], p=2, dim=-1), F.normalize(target_proto, p=2, dim=-1))
                    if self.args.data_dropout:
                        logits = []
                        estimated_proto = []
                        for jj in range(3):
                            if jj == 0:
                                a, b = model.module._forward(support, query, s_l, data_drop=False)
                            else:
                                a, b = model.module._forward(support, query, s_l)
                            logits.append(a)
                            estimated_proto.append(b)

                        loss = 0

                        for jj, (logit, e_p) in enumerate(zip(logits, estimated_proto)):
                            loss = loss + F.cross_entropy(logit, q_l)
                            estimated_proto[jj] = F.normalize(e_p, p=2, dim=-1)

                        dropcontrastive_loss = 0
                        for jj in range(5):
                            dropcontrastive_loss = dropcontrastive_loss + torch.dot(estimated_proto[0][jj], estimated_proto[1][jj]) + torch.dot(estimated_proto[0][jj], estimated_proto[2][jj]) + torch.dot(estimated_proto[1][jj], estimated_proto[2][jj])

                        #target_proto = []
                        #for i,l in enumerate(true_label[:5]):
                        #    tmp = torch.load(f'./full_data_proto/class{l}_rot{selected_rots[i]}.pt')
                        #    target_proto.append(tmp)
                        #target_proto = torch.stack(target_proto)
                        #proto_loss = abs(target_proto.cuda() - estimated_proto).mean()

                        total_loss = loss - dropcontrastive_loss * self.args.dropcontrastive_lamb# + proto_loss * self.args.proto_lamb
                        acc = count_acc(logits[0], q_l)

                    else:
                        logits, estimated_proto = model.module._forward(support, query, s_l)

                        #
                        target_proto = []
                        for i,l in enumerate(true_label[:5]):
                            tmp = torch.load(f'./full_data_proto/class{l}_rot{selected_rots[i]}.pt')
                            target_proto.append(tmp)
                        target_proto = torch.stack(target_proto)
                        proto_loss = abs(target_proto.cuda() - estimated_proto).mean()

                        total_loss = F.cross_entropy(logits, q_l) + proto_loss * self.args.proto_lamb
                        acc = count_acc(logits, q_l)

                lrc = scheduler.get_last_lr()[0]
                tqdm_gen.set_description(
                    'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
                tl.add(total_loss.item())
                ta.add(acc)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        tl = tl.item()
        ta = ta.item()
        return tl, ta

    def test(self, model, testloader, args, session):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        if session > 0:
            va_base = Averager()
            va_new = Averager()
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]

                if args.meta:
                    model.module.mode = 'encoder'
                    features = model(data)
                    logits = model.module._forward(features, None, test=True)
                    
                else:
                    model.module.mode = 'encoder'
                    query = model(data)
                    query = query.unsqueeze(0).unsqueeze(0)

                    proto = model.module.fc.weight[:test_class, :].detach()
                    proto = proto.unsqueeze(0).unsqueeze(0)

                    logits = model.module._forward(proto, query)

                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
                if session > 0:
                    if i >= 60:
                        va_new.add(acc)
                    else:
                        va_base.add(acc)
                
                if args.save_fig:
                    if i % 10 == 1:
                        class_idxes = np.arange(0,100)
                        p = logits[:5,:].clone().detach().cpu().numpy()

                        for j in range(5):
                            plt.ylim(-16,16)
                            plt.bar(class_idxes, p[j])
                            plt.xticks(np.arange(0,100,5))
                            plt.grid(True, axis='x', linestyle='--', alpha=0.4)
                            max_idx = np.argmax(p[j])
                            tmp_arr = np.zeros(100)
                            tmp_arr[max_idx] = p[j][max_idx]
                            plt.bar(class_idxes, tmp_arr, color='red')
                            plt.title('Total Scores')

                            os.makedirs(args.save_path+'/Figures/Class{}/Example{}'.format(i,j), exist_ok=True)
                            plt.tight_layout()
                            plt.savefig(args.save_path+'/Figures/Class{}/Example{}/Session{}.png'.format(i,j,session))
                            plt.cla()
                            plt.clf()

            vl = vl.item()
            va = va.item()
            if session > 0:
                va_base = va_base.item()
                va_new = va_new.item()
        
        if session == 0:
            return vl, va
        else:
            return vl, va, va_base, va_new

    def plot_cos_sim(self, session, tensor=None):
        if tensor is None:
            target_protos = 60 + session * 5
            target_protos = self.model.module.fc.weight.data[:target_protos]
            target_protos = F.normalize(target_protos, p=2, dim=-1)

            num_protos = 60 + session * 5 + 1
        else:
            target_protos = F.normalize(tensor)
            num_protos = tensor.shape[0]

        cos_sim = F.linear(target_protos, target_protos).detach().cpu().numpy()

        import matplotlib.pyplot as plt

        plt.imshow(cos_sim)
        #plt.xticks(np.arange(0,60 + session * 5+1,5))
        
        plt.yticks(np.arange(0,num_protos,5))
        plt.colorbar()
        os.makedirs(f'{self.args.save_path}/cossim', exist_ok=True)
        plt.savefig(f'{self.args.save_path}/cossim/session{session}.png')
        plt.cla()
        plt.clf()

    def set_save_path(self):

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        if self.args.save is not None:
            self.args.save_path = self.args.save_path + self.args.save

        else:
            self.args.save_path = self.args.save_path + '%dW-%dS-%dQ-%dEpi-L%dW-L%dS' % (
                self.args.episode_way, self.args.episode_shot, self.args.episode_query, self.args.train_episode,
                self.args.low_way, self.args.low_shot)
            # if self.args.use_euclidean:
            #     self.args.save_path = self.args.save_path + '_L2/'
            # else:
            #     self.args.save_path = self.args.save_path + '_cos/'
            if self.args.schedule == 'Milestone':
                mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
                self.args.save_path = self.args.save_path + 'Epo_%d-Lr1_%.6f-Lrg_%.5f-MS_%s-Gam_%.2f-T_%.2f' % (
                    self.args.epochs_base, self.args.lr_base, self.args.lrg, mile_stone, self.args.gamma,
                    self.args.temperature)
            elif self.args.schedule == 'Step':
                self.args.save_path = self.args.save_path + 'Epo_%d-Lr1_%.6f-Lrg_%.5f-Step_%d-Gam_%.2f-T_%.2f' % (
                    self.args.epochs_base, self.args.lr_base, self.args.lrg, self.args.step, self.args.gamma,
                    self.args.temperature)
    
            if 'ft' in self.args.new_mode:
                self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                    self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None

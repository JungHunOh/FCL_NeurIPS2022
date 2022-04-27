# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data)
        if False:#'trans' in args.new_mode:
            original = logits[1]
            logits = logits[0]
            
            logits = logits[:, :args.base_class]
            original = original[:, :args.base_class]

            loss = F.cross_entropy(logits, train_label) + F.cross_entropy(original, train_label)
        else:
            logits = logits[:, :args.base_class]
            loss = F.cross_entropy(logits, train_label)

        acc = count_acc(logits, train_label)
        
        fc = model.module.fc.weight[:60]
        fc = F.normalize(fc, p=2, dim=-1)
        fc_loss = F.linear(fc,fc)

        fc_loss = fc_loss.mean()

        total_loss = loss + fc_loss

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


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            if 'trans' in args.new_mode:
                affine_param = model.module.transformation_layer(embedding)
                embedding = embedding * affine_param[:,0].unsqueeze(1) + affine_param[:,1].unsqueeze(1)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model


def test(model, testloader, epoch, args, session, save_fig=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    if session > 0:
        va_base = Averager()
        va_new = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            if False:#'trans' in args.new_mode:
                logits = logits[0]
            logit = logits[:, :test_class]
            loss = F.cross_entropy(logit, test_label)
            acc = count_acc(logit, test_label)

            vl.add(loss.item())
            va.add(acc)

            if session > 0:
                if i >= 60:
                    va_new.add(acc)
                else:
                    va_base.add(acc)
            
            if save_fig:
                if i % 10 == 1:
                    class_idxes = np.arange(0,100)
                    if 'nph' in args.new_mode:
                        p = logits[0][:5,:].clone().detach().cpu().numpy() # [10,100]
                        n = logits[1][:5,:].clone().detach().cpu().numpy() # [10,100]
                    else:
                        p = logits[:5,:].clone().detach().cpu().numpy()

                    for j in range(5):
                        if 'nph' in args.new_mode:
                            plt.subplot(3,1,1)
                            plt.ylim(-16,16)
                            plt.bar(class_idxes, p[j])
                            plt.xticks(np.arange(0,100,5))
                            plt.grid(True, axis='x', linestyle='--', alpha=0.4)
                            plt.title('Positive Scores')
                            plt.subplot(3,1,2)
                            plt.ylim(-16,16)
                            plt.bar(class_idxes, n[j]*args.nph_lamb)
                            plt.xticks(np.arange(0,100,5))
                            plt.grid(True, axis='x', linestyle='--', alpha=0.4)
                            plt.title('Negative Scores')
                            plt.subplot(3,1,3)
                            plt.ylim(-16,16)
                            plt.bar(class_idxes, p[j]-n[j]*args.nph_lamb)
                            plt.xticks(np.arange(0,100,5))
                            plt.grid(True, axis='x', linestyle='--', alpha=0.4)
                            max_idx = np.argmax(p[j]-n[j] * args.nph_lamb)
                            tmp_arr = np.zeros(100)
                            tmp_arr[max_idx] = (p[j]-n[j] * args.nph_lamb)[max_idx]
                            plt.bar(class_idxes, tmp_arr,color='red')
                            plt.title('Total Scores')
                        else:
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
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        return vl, va
    else:
        print('epo {}, test, loss={:.4f} total_acc={:.4f}, base_acc={:.4f}, new_acc={:.4f}'.format(epoch, vl, va, va_base, va_new))
        return vl, va, va_base, va_new


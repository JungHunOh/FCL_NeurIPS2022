import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.Network import MYNET as Net
import numpy as np

class MYNET(Net):

    def __init__(self, args, mode=None):
        super().__init__(args,mode)

        self.args = args

        hdim=self.num_features
        if not args.meta:
            self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        else:
            #self.proto_estimator = ProtoEstimator(args)
            self.proto_estimator = MultiHeadAttention(self.args.num_heads, hdim, hdim, hdim, dropout=0, estimator=True)
            self.proto_refiner = MultiHeadAttention(self.args.num_heads, hdim, hdim, hdim, dropout=0, estimator=False)
            self.refined_fc = nn.Linear(64,100,bias=False)

            #self.proto_estimator = AttentionProtoEstimator(1, hdim, hdim, hdim)
            if self.args.feat_transform:
                #self.feat_transformer = FeatTransformer(args)
                self.feat_transformer = MultiHeadAttention(self.args.num_heads, hdim, hdim, hdim, dropout=0)
            if self.args.meta_sgd:
                self.meta_sgd_params = nn.ParameterList([nn.Parameter(torch.ones(param.shape) * self.args.inner_lr, requires_grad=True) for param in self.proto_estimator.parameters()])

    def forward(self, input):
        if self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            if self.args.meta:
                import pdb; pdb.set_trace
            else:
                support_idx, query_idx = input
                logits = self._forward(support_idx, query_idx)
                return logits

    def estimate_prototype(self, dataloader, class_list, session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        avg_feat = data.view(self.args.low_way, self.args.low_shot, data.shape[-1]) # [way, shot, feat]
        avg_feat = F.normalize(avg_feat, p=2, dim=-1)
        avg_feat = avg_feat.mean(dim=1)
        #avg_feat = F.normalize(avg_feat, p=2, dim=-1)

        data = data.view(self.args.low_way, self.args.low_shot, data.shape[-1]) # [way, shot, feat]

        for w in range(self.args.low_way):
            fc_weight = self.fc.weight[:class_list.min()+w].clone()

            target_data = data[w,:,:]
            target_data = F.normalize(target_data, p=2, dim=-1)
            target_label = label[w*5:(w+1)*5]

            if self.args.feat_transform:
                transformed_feat = self.feat_transformer(target_data)

            inner_params = {name: param for name, param in self.proto_estimator.named_parameters()}

            for i in range(self.args.inner_steps):
                #estimated_proto = self.proto_estimator(transformed_feat, inner_params).squeeze()# + avg_feat[w] # [feat]
                estimated_proto = self.proto_estimator(target_data, inner_params).squeeze()# + avg_feat[w] # [feat]


                cur_fc = torch.cat([fc_weight, estimated_proto.unsqueeze(0)], dim=0)
                cur_fc = self.proto_refiner(F.normalize(cur_fc, p=2, dim=-1))
                #fc_weight = self.fc.weight[:class_list.min()+w+1].clone()
                #fc_weight[-1] = estimated_proto

                #fc_weight = self.fc.weight.data.clone()
                #fc_weight[class_list.min()+w] = estimated_proto

                if self.args.feat_transform:
                    output = self.args.temperature * F.linear(F.normalize(transformed_feat, p=2, dim=-1), F.normalize(cur_fc, p=2, dim=-1))
                else:
                    output = self.args.temperature * F.linear(F.normalize(target_data, p=2, dim=-1), F.normalize(cur_fc, p=2, dim=-1))

                #contrastive_loss = F.linear(F.normalize(estimated_proto.unsqueeze(0), p=2, dim=-1), F.normalize(cur_fc[:class_list.min()+w], p=2, dim=-1)).mean()
                loss = F.cross_entropy(output, target_label)# + contrastive_loss * self.args.lamb_contrastive
                #self.zero_grad()
                #self.proto_estimator.zero_grad()
                grads = torch.autograd.grad(loss, inner_params.values())

                if self.args.meta_sgd:
                    updated_weights = list(map(
                            lambda current_params, lr, grads: current_params.to('cuda') -
                                                                          lr *
                                                                           (grads.to('cuda')),
                            inner_params.values(), self.meta_sgd_params, grads))
                else:
                    updated_weights = list(map(
                        lambda current_params, grads: current_params.to('cuda') -
                                                                      self.args.inner_lr *
                                                                       (grads.to('cuda')),
                        inner_params.values(), grads))

                inner_params = dict(zip(inner_params.keys(), updated_weights))

            #estimated_proto = self.proto_estimator(transformed_feat, inner_params).squeeze()# + avg_feat[w]
            estimated_proto = self.proto_estimator(target_data, inner_params).squeeze()# + avg_feat[w]
            #cur_fc = torch.cat([fc_weight, estimated_proto.unsqueeze(0)], dim=0)
            #cur_fc = self.proto_refiner(cur_fc)

            #estimated_proto = transformed_feat.mean(dim=0)
            #estimated_proto = avg_feat[w]

            self.fc.weight.data[class_list.min()+w] = estimated_proto
        
        cur_fc = self.proto_refiner(F.normalize(self.fc.weight[:class_list.min()+5], p=2, dim=-1))
        
        self.refined_fc.weight.data[:class_list.min()+5] = cur_fc

    def _forward(self,support,query,support_label=None,pseudo_proto=None,test=False,data_drop=True):
        if self.args.meta:
            if test:
                output = self.args.temperature * F.linear(F.normalize(support, p=2, dim=-1), F.normalize(self.refined_fc.weight, p=2, dim=-1))
                return output
            else:
                support_original, support_rot = support
                query_original, query_rot = query

                avg_feat = support_rot.view(self.args.low_shot, self.args.low_way, support_original.shape[-1]) # [shot, way, feat]
                avg_feat = F.normalize(avg_feat, p=2, dim=-1)
                avg_feat = avg_feat.mean(dim=0)
                #avg_feat = F.normalize(avg_feat, p=2, dim=-1)

                support_rot = support_rot.view(self.args.low_shot, self.args.low_way, support_original.shape[-1]) # [shot, way, feat]

                num_pseudo_new_classes = pseudo_proto.shape[0] if pseudo_proto is not None else 0

                #adapted_fc = self.fc.weight.data[:60+num_pseudo_new_classes+self.args.low_way].clone()
                adapted_fc = []

                if self.args.feat_transform:
                    transformed_feats = []

                fc_weight = self.fc.weight[:60].clone()
                if num_pseudo_new_classes > 0:
                    #fc_weight[60:60+num_pseudo_new_classes] = pseudo_proto
                    fc_weight = torch.cat([fc_weight, pseudo_proto], dim=0)

                for w in range(self.args.low_way):
                    target_rot = support_rot[:,w,:] # [shot, feat]
                    target_rot = F.normalize(target_rot, p=2, dim=-1)
                    #target_rot = self.feat_transformer(target_rot)
                    if self.args.feat_transform:
                        transformed_feat = self.feat_transformer(target_rot)
                        transformed_feats.append(transformed_feat)

                    if self.args.data_dropout and data_drop:
                        num = np.random.choice(self.args.low_shot-1,1) + 1
                        selected_data = np.sort(np.random.choice(5,num,replace=False))
                        data = target_rot[selected_data]

                        res_feat = data.mean(dim=0)
                    else:
                        data = target_rot
                        res_feat = avg_feat[w]
                        #res_feat = data.mean(dim=0)

                    target_label = support_label[0::self.args.low_way] # target label is always 60

                    inner_params = {name: param for name, param in self.proto_estimator.named_parameters()}

                    for i in range(self.args.inner_steps):
                        #estimated_proto = self.proto_estimator(transformed_feat, inner_params).squeeze()# + res_feat # [feat]
                        estimated_proto = self.proto_estimator(data, inner_params).squeeze()# + res_feat # [feat]

                        '''
                        fc_weight = self.fc.weight[:60+num_pseudo_new_classes+1].clone()
                        if num_pseudo_new_classes > 0:
                            fc_weight[60:60+num_pseudo_new_classes] = pseudo_proto
                            #fc_weight = torch.cat([fc_weight, pseudo_proto])
                        fc_weight[-1] = estimated_proto
                        '''
                        cur_fc = torch.cat([fc_weight, estimated_proto.unsqueeze(0)], dim=0)

                        cur_fc = self.proto_refiner(F.normalize(cur_fc,p=2,dim=-1))

                        #output = self.args.temperature * F.linear(F.normalize(torch.cat([support_original, support_rot], dim=0), p=2, dim=-1), F.normalize(fc_weight, p=2, dim=-1))

                        #output = self.args.temperature * F.linear(F.normalize(target_rot, p=2, dim=-1), F.normalize(fc_weight, p=2, dim=-1))
                        if self.args.feat_transform:
                            output = self.args.temperature * F.linear(F.normalize(transformed_feat, p=2, dim=-1), F.normalize(cur_fc, p=2, dim=-1))
                        else:
                            output = self.args.temperature * F.linear(F.normalize(target_rot, p=2, dim=-1), F.normalize(cur_fc, p=2, dim=-1))

                        #1contrastive_loss = F.linear(F.normalize(estimated_proto.unsqueeze(0), p=2, dim=-1), F.normalize(cur_fc[:60+num_pseudo_new_classes], p=2, dim=-1)).mean()
                        
                        loss = F.cross_entropy(output, target_label)
                        #if w == 0:
                        #    print(f'###{round(loss.item(),3)}, {round(contrastive_loss.item(),3)}###')
                        #loss = loss + contrastive_loss * self.args.lamb_contrastive
                        #self.proto_estimator.zero_grad()
                        grads = torch.autograd.grad(loss, inner_params.values(), create_graph=self.args.second_order_maml)
                        #grads = torch.autograd.grad(loss, inner_params.values(), create_graph=self.args.second_order_maml, retain_graph=True)
                        if self.args.meta_sgd:
                            updated_weights = list(map(
                                    lambda current_params, lr, grads: current_params.to('cuda') -
                                                                                  lr *
                                                                                   (grads.to('cuda')),
                                    inner_params.values(), self.meta_sgd_params, grads))
                        else:
                            updated_weights = list(map(
                                lambda current_params, grads: current_params.to('cuda') -
                                                                              self.args.inner_lr *
                                                                               (grads.to('cuda')),
                                    inner_params.values(), grads))

                        inner_params = dict(zip(inner_params.keys(), updated_weights))

                    #estimated_proto = self.proto_estimator(transformed_feat, inner_params)# + res_feat # [feat]
                    estimated_proto = self.proto_estimator(data, inner_params)# + res_feat # [feat]
                    
                    cur_fc = torch.cat([fc_weight, estimated_proto.unsqueeze(0)], dim=0)
                    cur_fc = self.proto_refiner(F.normalize(cur_fc, p=2, dim=-1))

                    #estimated_proto = res_feat # [feat]
                    #estimated_proto = transformed_feat.mean(dim=0)

                    adapted_fc.append(cur_fc)

                #fc_weight = self.fc.weight.data[60:60+estimated_proto.shape[0]].clone()
                #fc_weight = estimated_proto

                #output = self.args.temperature * F.linear(F.normalize(torch.cat([query_original, query_rot], dim=0), p=2, dim=-1), F.normalize(adapted_fc, p=2, dim=-1))
                output = None
                #output = self.args.temperature * F.linear(F.normalize(query_rot, p=2, dim=-1), F.normalize(fc_weight, p=2, dim=-1))
                if self.args.feat_transform:
                    return output, adapted_fc, transformed_feats
                else:
                    return output, adapted_fc, None

        else:
            emb_dim = support.size(-1)
            # get mean of the support
            proto = support.mean(dim=1)
            num_batch = proto.shape[0]
            num_proto = proto.shape[1]
            num_query = query.shape[1]*query.shape[2]#num of query*way

            # query: (num_batch, num_query, num_proto, num_emb)
            # proto: (num_batch, num_proto, num_emb)
            query = query.view(-1, emb_dim).unsqueeze(1)

            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim)

            combined = torch.cat([proto, query], 1) # Nk x (N + 1) x d, batch_size = NK
            combined = self.slf_attn(combined, combined, combined)
            # compute distance for all batches
            proto, query = combined.split(num_proto, 1)

            logits=F.cosine_similarity(query,proto,dim=-1)
            logits=logits*self.args.temperature

        return logits


class FeatTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fc_1 = nn.Linear(64,64)
        self.fc_2 = nn.Linear(64,2)

    def forward(self, x, params=None):
        # x : [shot, 64]

        original = x
        if params is None:
            x = F.relu(self.fc_1(x))
            x = self.fc_2(x)
        else:
            x = F.relu(F.linear(x, params['fc_1.weight'], bias=params['fc_1.bias']))
            x = F.linear(x, params['fc_2.weight'], bias=params['fc_2.bias'])

        gammas = x[:,0]
        betas = x[:,1]
        x = original * gammas.unsqueeze(-1) + betas.unsqueeze(-1)

        return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v=None):
        
        attn = torch.bmm(q, k.transpose(1,2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, dim=-1)
        attn = self.softmax(attn)
        #attn = self.dropout(attn)
        if v is not None:
            output = torch.bmm(attn, v).transpose(0,1).reshape(v.shape[1], v.shape[2] * v.shape[0])
        else:
            output = None
        return output, attn, log_attn

class ProtoEstimator(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        
        #self.encoder = nn.Sequential(*[nn.Linear(64,32,bias=False), nn.ReLU(inplace=True), nn.Linear(32,16,bias=False)])
        #self.combiner = nn.Sequential(*[nn.Linear(16*5,16*5,bias=False), nn.ReLU(inplace=True), nn.Linear(16*5,64,bias=False)])

        self.fc1 = nn.Linear(64,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        #self.fc4 = nn.Linear(64,64)
        #self.dropout = nn.Dropout(0.1)

    def forward(self, data, params, test=False):
        # data [shot*way, feat]
        shot = self.args.low_shot
        way = self.args.low_way

        #avg -> fc

        data = data.mean(dim=0).unsqueeze(0)
        #residual = data.mean(dim=0).unsqueeze(0)
        x = F.relu(F.linear(data, params['fc1.weight'], bias=params['fc1.bias']))
        #x = self.dropout(x)
        x = F.relu(F.linear(x, params['fc2.weight'], bias=params['fc2.bias']))
        #x = self.dropout(x)
        #estimated_proto = F.relu(F.linear(x.mean(dim=0).unsqueeze(0), params['fc3.weight'], bias=params['fc3.bias']))
        estimated_proto = F.linear(x, params['fc3.weight'], bias=params['fc3.bias']) + data
        
        #x = F.relu(F.linear(data, params['fc1.weight'], bias=params['fc1.bias']))
        #x = F.relu(F.linear(x, params['fc2.weight'], bias=params['fc2.bias']))
        #x = F.relu(F.linear(x, params['fc3.weight'], bias=params['fc3.bias']))
        #estimated_proto = F.linear(x, params['fc4.weight'], bias=params['fc4.bias'])
        #estimated_proto = estimated_proto.mean(dim=0).unsqueeze(0)

        
        #encode and combine
        '''
        encoded = F.relu(F.linear(data, params['encoder.0.weight']))
        encoded = F.linear(encoded, params['encoder.2.weight'])
        '''
        '''
        if test:
            encoded = encoded.reshape(way, shot, encoded.shape[-1]).reshape(way, -1)
        else:
            encoded = encoded.view(shot, way, encoded.shape[-1]) # [shot, way, feat]
            encoded = encoded.transpose(0,1).reshape(encoded.shape[1], -1) # [way, shot * feat]
        '''
        '''
        encoded = encoded.view(-1)

        estimated_proto = F.relu(F.linear(encoded, params['combiner.0.weight']))
        estimated_proto = F.linear(estimated_proto, params['combiner.2.weight'])
        
        '''
        #return F.normalize(estimated_proto, p=2, dim=-1)
        return estimated_proto.squeeze()


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0, estimator=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.estimator = estimator

        #self.pre_fc1 = nn.Linear(d_model, d_model)
        #self.pre_fc2 = nn.Linear(d_model, d_model)

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        #self.w_qs2 = nn.Linear(d_model, n_head * d_k, bias=False)
        #self.w_ks2 = nn.Linear(d_model, n_head * d_k, bias=False)
        #self.w_vs2 = nn.Linear(d_model, n_head * d_v, bias=False)
        #nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        #self.layer_norm2 = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        #self.fc2 = nn.Linear(n_head * d_v, d_model)
        #nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, params=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        '''
        residual = q

        if self.estimator:
            q = F.relu(F.linear(q, params['pre_fc1.weight'], bias=params['pre_fc1.bias']))
            q = F.linear(q, params['pre_fc2.weight'], bias=params['pre_fc2.bias'])
        else:
            q = F.relu(self.pre_fc1(q))
            q = self.pre_fc2(q)
        
        q = q + residual
        '''
        residual = q
        
        qq = q.clone()
        kk = q.clone()
        vv = q.clone()

        if self.estimator:
            q = F.linear(qq, params['w_qs.weight'])
            k = F.linear(kk, params['w_ks.weight'])
            v = F.linear(vv, params['w_vs.weight'])
        else:
            q = self.w_qs(qq)
            k = self.w_ks(kk)
            v = self.w_vs(vv)

        q = q.view(q.shape[0], n_head, d_k).transpose(0,1)
        k = k.view(k.shape[0], n_head, d_k).transpose(0,1)
        v = v.view(v.shape[0], n_head, d_k).transpose(0,1)

        output, attn, log_attn = self.attention(q, k, v)

        if self.estimator:
            output = F.linear(output, params['fc.weight']) + residual
        else:
            output = self.fc(output)+residual

        if self.estimator:
            output = output.mean(dim=0)

        #output = self.layer_norm(output + residual)
        '''
        residual = output

        qq = output.clone()
        kk = output.clone()
        vv = output.clone()

        q = self.w_qs2(qq)
        k = self.w_ks2(kk)
        v = self.w_vs2(vv)

        output, attn, log_attn = self.attention(q, k, v)

        output = self.fc2(output)

        output = self.layer_norm2(output + residual)
        '''

        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        '''
        return output


class AttentionProtoEstimator(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        #nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=0)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

        self.fc = nn.Linear(n_head * d_v, d_model)
        #nn.init.xavier_normal_(self.fc.weight)
        #self.dropout = nn.Dropout(0.1)

    def forward(self, qq, params):
        #q = F.normalize(qq, p=2, dim=-1)
        #k = F.normalize(qq, p=2, dim=-1)
        #v = F.normalize(qq, p=2, dim=-1)

        q = qq.clone().unsqueeze(0)
        k = qq.clone().unsqueeze(0)
        v = qq.clone().unsqueeze(0)
        
        residual = q

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = F.linear(q, params['w_qs.weight']).view(sz_b, len_q, n_head, d_k)
        k = F.linear(k, params['w_ks.weight']).view(sz_b, len_k, n_head, d_k)
        v = F.linear(v, params['w_vs.weight']).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        #output = self.dropout(self.fc(output))
        output = F.linear(output, params['fc.weight'], bias=params['fc.bias'])#self.fc(output)
        output = F.layer_norm(output, (self.d_model,), weight=params['layer_norm.weight'], bias=params['layer_norm.bias'])#self.layer_norm(output + residual)
        
        return output.mean(dim=1).squeeze()

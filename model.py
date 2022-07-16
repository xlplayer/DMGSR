import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import math
import pandas as pd
import networkx as nx
import numpy as np
import copy
import dgl
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.utils import Identity
import dgl.nn.pytorch as dglnn
from label_smooth import LabelSmoothSoftmax


def udf_agg(edges):
    return {'m':edges.src['ft']*torch.sum(edges.data['e'] * edges.dst['ft'], dim=-1, keepdim=True)}


def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        elif any(map(lambda x: x in name, ['gating'])):
            decay.append(param)
        else:
            no_decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params



class GATLayer(nn.Module):
    def __init__(self, dim, num_heads, idx):
        super(GATLayer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.idx = idx

        self.pi =  nn.Linear(self.dim * self.num_heads, 1, bias=False)
        self.fc = nn.Linear(self.dim, self.dim*self.num_heads)
        self.feat_drop = nn.Dropout(config.feat_drop)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h_v, g):
        with g.local_scope():
            ###item to item
            adj = g.edge_type_subgraph(['interacts'+str(self.idx)])
            adj.nodes['item'].data['ft'] = self.fc(h_v)
            adj.apply_edges(fn.u_mul_v('ft','ft','e'), etype='interacts'+str(self.idx))
            e = self.pi(adj.edges['interacts'+str(self.idx)].data['e'])
            e = self.leaky_relu(e)

            adj.edges['interacts'+str(self.idx)].data['a'] = edge_softmax(adj['interacts'+str(self.idx)], e)
            adj.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'), etype='interacts'+str(self.idx))
            rst = adj.nodes['item'].data['ft']
            return torch.max(rst.view(-1,self.num_heads,config.dim), dim=1)[0]

class GAT(nn.Module):
    def __init__(self, dim, num_heads=8, idx=0):
        super(GAT, self).__init__()
        self.layer1 = GATLayer(dim, num_heads = num_heads, idx=idx)

    def forward(self, h_0, g):
        h1 = self.layer1(h_0, g)
        return h_0+h1
       


# pylint: enable=W0235
class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_d = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_d, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            # ed = (d_feat.view(-1, self._num_heads, self._out_feats) *self.attn_d).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, -1, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class PosAggregator(nn.Module):
    def __init__(self, dim, last_L=1):
        super(PosAggregator, self).__init__()
        self.dim = dim

        self.q = nn.Linear(2*self.dim, self.dim, bias=False)
        self.r = nn.Linear(2*self.dim, self.dim, bias=False)
        self.GRU = nn.GRU(self.dim, self.dim, 1, True, True)
        self.last_L = last_L


    def forward(self, h_v, h_p, h_t, g):
        with g.local_scope():
            adj = g.edge_type_subgraph(['agg'])
            adj.nodes['item'].data['ft'] = h_v
            adj.nodes['target'].data['ft'] = h_t
            adj.edges['agg'].data['pos'] = h_p
            adj.apply_edges(fn.copy_src('ft','ft'))
            e = self.q(torch.cat([adj.edata['ft'], adj.edata['pos']], dim=-1))
            adj.edata['e'] = torch.tanh(e)

            last_nodes = adj.filter_nodes(lambda nodes: nodes.data['last'][:,0]==1, ntype='item')
            last_feat = adj.nodes['item'].data['ft'][last_nodes]
            last_feat = last_feat.unsqueeze(1).repeat(1,1,1).view(-1, config.dim)

            # f = self.r(torch.cat([adj.nodes['target'].data['ft'], last_feat], dim=-1))
            # adj.nodes['target'].data['ft'] = f
            adj.update_all(udf_agg, fn.sum('m', 'ft'))

            return g.nodes['target'].data['ft'], last_feat


class HardSession(nn.Module):
    def __init__(self, num_node, feat_drop=config.feat_drop, num_heads=8, density=3, pos_embedding=None, target_embedding = None, saparated=True, share=True):
        super(HardSession, self).__init__()
        self.num_node = num_node
        self.density = density
        self.saparated = saparated
        self.embedding = nn.Embedding(self.num_node, config.dim)

        if share:
            self.pos_embedding = pos_embedding
            self.target_embedding = target_embedding
        else:
            self.pos_embedding = nn.Embedding(200, config.dim)
            self.target_embedding = nn.Embedding(10, config.dim)
        self.feat_drop = nn.Dropout(feat_drop)
        
        self.gat1   = nn.ModuleList()
        self.gat2 = nn.ModuleList()
        self.agg = nn.ModuleList()
        self.fc_sr = nn.ModuleList()
        self.sc_sr = nn.Sequential(nn.Linear(config.dim, config.dim, bias=True),  nn.ReLU(), nn.Linear(config.dim, 2, bias=False), nn.Softmax(dim=-1))
        for i in range(self.density):
            self.gat1.append(dglnn.HeteroGraphConv({"interacts"+str(i):GATConv(config.dim, config.dim, num_heads, feat_drop, feat_drop, residual=True, allow_zero_in_degree=True)}, aggregate='sum'))
            self.gat2.append(dglnn.HeteroGraphConv({"interacts"+str(i):GATConv(config.dim, config.dim, num_heads, feat_drop, feat_drop, residual=True, allow_zero_in_degree=True)}, aggregate='sum'))
            self.agg.append(PosAggregator(config.dim, i+1))
            self.fc_sr.append(nn.Linear(2*config.dim, config.dim, bias=False))
    
    def forward(self, g, targets, epoch=None, training=False):
        h_v = self.embedding(g.nodes['item'].data['iid'])
        h_v = self.feat_drop(h_v)
        h_v = F.normalize(h_v, dim=-1)

        h_p = self.pos_embedding(g.edges['agg'].data['pid'])
        h_r = self.target_embedding(g.nodes['target'].data['tid'])

        feat, last_feat = [],[]
        for i in range(self.density):
            ##0nd
            q,p = self.agg[i](h_v, h_p, h_r, g)
            ##
            h1 = self.gat1[i](g.edge_type_subgraph(['interacts'+str(i)]), {'item':h_v})
            h1 = torch.max(h1['item'], dim=1)[0]
            h2 = self.gat2[i](g.reverse(copy_edata=True).edge_type_subgraph(['interacts'+str(i)]), {'item':h_v})
            h2 = torch.max(h2['item'], dim=1)[0]
            h = h1+h2
            h = F.normalize(h, dim=-1)
            x, y = self.agg[i](h, h_p, h_r, g)

            ##2nd
            h1 = self.gat1[i](g.edge_type_subgraph(['interacts'+str(i)]), {'item':h})
            h1 = torch.max(h1['item'], dim=1)[0]
            h2 = self.gat2[i](g.reverse(copy_edata=True).edge_type_subgraph(['interacts'+str(i)]), {'item':h})
            h2 = torch.max(h2['item'], dim=1)[0]
            h = h1+h2
            h = F.normalize(h, dim=-1)

            z, w = self.agg[i](h, h_p, h_r, g)
            ##
            x = q+x+z
            y = p+y+w

            feat.append(x.unsqueeze(1))
            last_feat.append(y.unsqueeze(1))
        
        sr_g = torch.cat(feat, dim=1)
        sr_l = torch.cat(last_feat, dim=1)
        sr   = torch.cat([sr_l, sr_g], dim=-1)
        sr   = torch.cat([self.fc_sr[i](sr).unsqueeze(1) for i, sr in enumerate(torch.unbind(sr, dim=1))], dim=1)

        sr = F.normalize(sr, dim=-1)
        b = self.embedding.weight
        b = F.normalize(b, dim=-1)

        logits = sr @ b.t()

        if self.saparated:
            phi = self.sc_sr(sr).unsqueeze(-1)
            mask = torch.zeros(phi.size(0), config.num_node).cuda()
            iids = torch.split(g.nodes['item'].data['iid'], g.batch_num_nodes('item').tolist())
            for i in range(len(mask)):
                mask[i, iids[i]] = 1

            logits_in = logits.masked_fill(~mask.bool().unsqueeze(1), float('-inf'))
            logits_ex = logits.masked_fill(mask.bool().unsqueeze(1), float('-inf'))
            score     = torch.softmax(12 * logits_in, dim=-1)
            score_ex  = torch.softmax(12 * logits_ex, dim=-1)
            score = (torch.cat((score.unsqueeze(2), score_ex.unsqueeze(2)), dim=2) * phi).sum(2)

        else:
            score = torch.softmax(12 * logits, dim=-1)

        score = score.mean(1)

        if not training:
            return score,sr_l

        score = torch.log(score)
        loss = self.loss_function(score, targets)
        return loss
        
class EasySession(nn.Module):
    def __init__(self, num_node, feat_drop=config.feat_drop, num_heads=8, density=3, pos_embedding=None, target_embedding = None, saparated=False, share=True):
        super(EasySession, self).__init__()
        self.num_node = num_node
        self.density = density
        self.saparated = saparated
        self.embedding = nn.Embedding(self.num_node, config.dim)

        if share:
            self.pos_embedding = pos_embedding
            self.target_embedding = target_embedding
        else:
            self.pos_embedding = nn.Embedding(200, config.dim)
            self.target_embedding = nn.Embedding(10, config.dim)
        self.feat_drop = nn.Dropout(feat_drop)
        self.gat1   = nn.ModuleList()
        self.gat2 = nn.ModuleList()
        self.agg = nn.ModuleList()
        self.fc_sr = nn.ModuleList()
        self.sc_sr = nn.Sequential(nn.Linear(config.dim, config.dim, bias=True),  nn.ReLU(), nn.Linear(config.dim, 2, bias=False), nn.Softmax(dim=-1))
        for i in range(self.density):
            self.gat1.append(GAT(config.dim, num_heads=num_heads, idx=i))
            self.gat2.append(GAT(config.dim, num_heads=num_heads, idx=i))
            self.agg.append(PosAggregator(config.dim))
            self.fc_sr.append(nn.Linear(2*config.dim, config.dim, bias=False))
        
    
    def forward(self, g, targets, epoch=None, training=False):
        h_v = self.embedding(g.nodes['item'].data['iid'])
        h_v = self.feat_drop(h_v)
        h_v = F.normalize(h_v, dim=-1)
        
        h_p = self.pos_embedding(g.edges['agg'].data['pid'])
        h_r = self.target_embedding(g.nodes['target'].data['tid'])

        feat, last_feat = [],[]
        for i in range(self.density):
            ##0nd
            q,p = self.agg[i](h_v, h_p, h_r, g)
            ##
            h1 =self.gat1[i](h_v, g)
            h2 = self.gat2[i](h_v, g.reverse(copy_edata=True).edge_type_subgraph(['interacts'+str(i)]))
            h = h1+h2
            h = F.normalize(h, dim=-1)

            x, y = self.agg[i](h, h_p, h_r, g)


            ##2nd
            h1 =self.gat1[i](h, g)
            h2 = self.gat2[i](h, g.reverse(copy_edata=True).edge_type_subgraph(['interacts'+str(i)]))
            h = h1+h2
            h = F.normalize(h, dim=-1)

            z, w = self.agg[i](h, h_p, h_r, g)
            ##
            x = q+x+z
            y = p+y+w

            feat.append(x.unsqueeze(1))
            last_feat.append(y.unsqueeze(1))

       
        sr_g = torch.cat(feat, dim=1)
        sr_l = torch.cat(last_feat, dim=1)
        sr   = torch.cat([sr_l, sr_g], dim=-1)
        sr   = torch.cat([self.fc_sr[i](sr).unsqueeze(1) for i, sr in enumerate(torch.unbind(sr, dim=1))], dim=1)

        sr = F.normalize(sr, dim=-1)
        b = self.embedding.weight
        b = F.normalize(b, dim=-1)

        logits = sr @ b.t()

        if self.saparated:
            phi = self.sc_sr(sr).unsqueeze(-1)
            mask = torch.zeros(phi.size(0), config.num_node).cuda()
            iids = torch.split(g.nodes['item'].data['iid'], g.batch_num_nodes('item').tolist())
            for i in range(len(mask)):
                mask[i, iids[i]] = 1

            logits_in = logits.masked_fill(~mask.bool().unsqueeze(1), float('-inf'))
            logits_ex = logits.masked_fill(mask.bool().unsqueeze(1), float('-inf'))
            score     = torch.softmax(12 * logits_in.squeeze(), dim=-1)
            score_ex  = torch.softmax(12 * logits_ex.squeeze(), dim=-1)
            score = (torch.cat((score.unsqueeze(2), score_ex.unsqueeze(2)), dim=2) * phi).sum(2)

        else:
            score = torch.softmax(12 * logits, dim=-1)
            
        score = score.mean(1)
        
        if not training:
            return score,sr_l

        score = torch.log(score)
        loss = self.loss_function(score, targets)
        return loss


class DMGSR(nn.Module):
    def __init__(self, num_node, feat_drop=config.feat_drop, num_heads=config.num_heads, density=config.density):
        super(DMGSR, self).__init__()
        self.num_node = num_node
        self.density = density

        self.pos_embedding = nn.Embedding(200, config.dim)
        self.target_embedding = nn.Embedding(10, config.dim)        
        self.gating = nn.Linear(2*self.density*config.dim,1,bias=False)

        self.esay = EasySession(num_node=num_node, feat_drop=feat_drop, num_heads=num_heads, density=density, pos_embedding=self.pos_embedding, target_embedding = self.target_embedding, saparated=False, share=True)
        self.hard = HardSession(num_node=num_node, feat_drop=feat_drop, num_heads=num_heads, density=density, pos_embedding=self.pos_embedding, target_embedding = self.target_embedding, saparated=True, share=True)
        
        self.loss_function = LabelSmoothSoftmax(lb_smooth=config.lb_smooth, reduction='mean')
        if config.weight_decay > 0:
            params = fix_weight_decay(self)
        else:
            params = self.parameters()
        self.optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_dc_step, gamma=config.lr_dc)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(config.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    
    def forward(self, g, targets=None, training=False):        
        score1,sr1 = self.esay(g, targets, training=False)
        score2,sr2 = self.hard(g, targets, training=False)

        gamma = self.gating(torch.cat([sr1.view(-1,self.density*config.dim), sr2.view(-1,self.density*config.dim)],dim=-1))
        gamma = torch.sigmoid(gamma)
        score = gamma*score1 + (1.-gamma)*score2
        if not training:
            return score
            
        loss = self.loss_function(torch.log(score1), targets) + self.loss_function(torch.log(score2), targets)
        
        
        score1 = score1.clone().detach()
        score2 = score2.clone().detach()
        gamma = self.gating(torch.cat([sr1.view(-1,self.density*config.dim), sr2.view(-1,self.density*config.dim)],dim=-1))
        gamma = torch.sigmoid(gamma)
        score = gamma*score1 + (1.-gamma)*score2
        
        epsilon = 1e-8
        kl_loss = score1 * (torch.log(score1+epsilon) - torch.log(score+epsilon)) + score2 * (torch.log(score2+epsilon) - torch.log(score+epsilon))
        regularization_loss  = torch.mean(torch.sum(kl_loss, dim=-1), dim=-1)
        loss += config.lambda_*regularization_loss

        return loss
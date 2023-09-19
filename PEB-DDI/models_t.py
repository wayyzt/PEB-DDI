import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (
                                TransformerConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool,
                                Linear,
                                )
from layers import (
                    RESCAL,
                    MergeFD,
                    CoAttentionLayer,
                    )
import time



class MVN_DDI(nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params, edge_feature,dp):
        super().__init__()
        self.in_node_features = in_node_features[0]
        self.in_node_features_fp = in_node_features[1]
        self.in_node_features_desc = in_node_features[2]
        self.in_edge_features = in_edge_features
        self.hidd_dim = hidd_dim
        self.kge_dim = kge_dim
        self.rel_total = rel_total
        self.n_blocks = len(blocks_params)
        
        self.initial_node_feature = Linear(self.in_node_features, self.hidd_dim ,bias=True, weight_initializer='glorot')
        self.initial_edge_feature = Linear(self.in_edge_features, edge_feature ,bias=True, weight_initializer='glorot')
        self.initial_node_norm = LayerNorm(self.hidd_dim)
        
        self.blocks = []
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = MVN_DDI_Block(self.hidd_dim, n_heads, head_out_feats, edge_feature, dp)
            # block = DeeperGCN(self.hidd_dim, n_heads, head_out_feats)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)
        self.fdmer = MergeFD(self.in_node_features_fp, self.in_node_features_desc, self.kge_dim)
        # self.fdmer = MergeFD_trans(self.in_node_features_fp, self.in_node_features_desc, self.kge_dim, "Transformer")
        self.merge_all = nn.Sequential(nn.Linear(self.kge_dim * 2, self.kge_dim),
                                   nn.ReLU())

    def forward(self, triples):
        h_data, h_data_fin, h_data_desc, t_data, t_data_fin, t_data_desc, rels, h_data_edge, t_data_edge = triples
        # h_data, h_data_fin, h_data_desc, t_data, t_data_fin, t_data_desc, rels = triples
        
        # 线性变换 55-64/128
        h_data.x = self.initial_node_feature(h_data.x)
        t_data.x = self.initial_node_feature(t_data.x)
        h_data.x = self.initial_node_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_node_norm(t_data.x, t_data.batch)
        h_data.x = F.elu(h_data.x)
        t_data.x = F.elu(t_data.x)

        h_data.edge_attr = self.initial_edge_feature(h_data.edge_attr)
        t_data.edge_attr = self.initial_edge_feature(t_data.edge_attr)
        h_data.edge_attr = F.elu(h_data.edge_attr)
        t_data.edge_attr = F.elu(t_data.edge_attr)

        
        # 4层gat 64-32*2
        repr_h = []
        repr_t = []
        for i, block in enumerate(self.blocks):
            out = block(h_data,t_data,h_data_edge, t_data_edge)
            # out = block(h_data,t_data,h_data_desc,t_data_desc)
            h_data = out[0]
            t_data = out[1]
            h_global_graph_emb = out[2]
            t_global_graph_emb = out[3]
            repr_h.append(h_global_graph_emb)
            repr_t.append(t_global_graph_emb)


        _, _, h_data_fin, h_data_desc, t_data_fin, t_data_desc = self.fdmer(h_data_fin,h_data_desc,t_data_fin,t_data_desc)
        repr_h_fd = []
        repr_t_fd = []
        for i in range(len(self.blocks)):
            repr_h_fd.append(F.normalize(repr_h[i] + h_data_fin))
            repr_t_fd.append(F.normalize(repr_t[i] + t_data_fin))
        
        repr_h = torch.stack((repr_h_fd), dim=1)
        repr_t = torch.stack((repr_t_fd), dim=1)
        kge_heads = repr_h #1024,4,128
        kge_tails = repr_t #1024,4,128
        attentions = self.co_attention(kge_heads, kge_tails)
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)

        return scores     


class MVN_DDI_Block(nn.Module):
    def __init__(self, in_features, n_heads, head_out_feats, edge_feature, dp):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.feature_conv = TransformerConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        self.lin_up = Linear(64, 64, bias=True, weight_initializer='glorot')
        
        self.feature_conv2 = TransformerConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        self.lin_up2 = Linear(64, 64, bias=True, weight_initializer='glorot')
        
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
        self.re_shape = Linear(64 + 128, 128, bias=True, weight_initializer='glorot')
        self.norm = LayerNorm(n_heads * head_out_feats)
        self.norm2 = LayerNorm(n_heads * head_out_feats)
        
        self.re_shape_e = Linear(64, 128, bias=True, weight_initializer='glorot')
    
    def forward(self, h_data,t_data,h_data_edge, t_data_edge):
        h_x_in = h_data.x
        t_x_in = t_data.x

        # node_update
        h_data, t_data = self.ne_update(h_data, t_data)

        # global
        h_data.x = self.feature_conv2(h_data.x, h_data.edge_index, h_data.edge_attr)
        t_data.x = self.feature_conv2(t_data.x, t_data.edge_index, t_data.edge_attr)
        h_data.edge_attr = self.lin_up2(h_data.edge_attr)
        t_data.edge_attr = self.lin_up2(t_data.edge_attr)

        h_global_graph_emb, t_global_graph_emb = self.GlobalPool(h_data, t_data, h_data_edge, t_data_edge)

        # node_shortcut
        h_data.x = F.elu(self.norm2(h_data.x, h_data.batch))
        t_data.x = F.elu(self.norm2(t_data.x, t_data.batch))
        
        h_data.edge_attr = F.elu(h_data.edge_attr)
        t_data.edge_attr = F.elu(t_data.edge_attr)

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb

    def ne_update(self, h_data, t_data):
        h_data.x = self.feature_conv(h_data.x, h_data.edge_index, h_data.edge_attr)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index, t_data.edge_attr)
        h_data.x = F.elu(self.norm(h_data.x, h_data.batch))
        t_data.x = F.elu(self.norm(t_data.x, t_data.batch))

        h_data.edge_attr = self.lin_up(h_data.edge_attr)
        t_data.edge_attr = self.lin_up(t_data.edge_attr)
        h_data.edge_attr = F.elu(h_data.edge_attr)
        t_data.edge_attr = F.elu(t_data.edge_attr)

        return h_data, t_data

    def GlobalPool(self, h_data, t_data,h_data_edge, t_data_edge):
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores= self.readout(h_data.x, h_data.edge_index, edge_attr=h_data.edge_attr, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores= self.readout(t_data.x, t_data.edge_index, edge_attr=t_data.edge_attr, batch=t_data.batch)
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)

        h_data_edge.x = h_data.edge_attr
        t_data_edge.x = t_data.edge_attr
        h_global_graph_emb_edge = global_add_pool(h_data_edge.x, batch=h_data_edge.batch)
        t_global_graph_emb_edge = global_add_pool(t_data_edge.x, batch=t_data_edge.batch)
        h_global_graph_emb_edge = F.elu(self.re_shape_e(h_global_graph_emb_edge))
        t_global_graph_emb_edge = F.elu(self.re_shape_e(t_global_graph_emb_edge))

        h_global_graph_emb = h_global_graph_emb * h_global_graph_emb_edge
        t_global_graph_emb = t_global_graph_emb * t_global_graph_emb_edge
        return h_global_graph_emb, t_global_graph_emb

    
import torch

from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (
                                GATConv,
                                TransformerConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool,
                                Linear,
                                )

from layers import (
                    CoAttentionLayer, 
                    RESCAL, 
                    IntraGraphAttention,
                    InterGraphAttention,
                    MergeFD,
                    )
import time




class MVN_DDI(nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params):
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
        self.initial_edge_feature = Linear(self.in_edge_features, 64 ,bias=True, weight_initializer='glorot')
        self.initial_node_norm = LayerNorm(self.hidd_dim)
        
        self.blocks = []
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = MVN_DDI_Block(self.hidd_dim, n_heads, head_out_feats)
            # block = DeeperGCN(self.hidd_dim, n_heads, head_out_feats)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples):
        h_data, t_data, rels, b_graph, h_data_fin, t_data_fin, h_data_edge, t_data_edge  = triples

        # 线性变换 55-64/128
        h_data.x = self.initial_node_feature(h_data.x)
        t_data.x = self.initial_node_feature(t_data.x)
        h_data.x = self.initial_node_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_node_norm(t_data.x, t_data.batch)
        h_data.x = F.relu(h_data.x)
        t_data.x = F.relu(t_data.x)

        h_data.edge_attr = self.initial_edge_feature(h_data.edge_attr)
        t_data.edge_attr = self.initial_edge_feature(t_data.edge_attr)
        h_data.edge_attr = F.relu(h_data.edge_attr)
        t_data.edge_attr = F.relu(t_data.edge_attr)

        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data,t_data,b_graph,h_data_edge, t_data_edge)

            h_data = out[0]
            t_data = out[1]
            h_global_graph_emb = out[2]
            t_global_graph_emb = out[3]
            repr_h.append(h_global_graph_emb)
            repr_t.append(t_global_graph_emb)
        
        
        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        kge_heads = repr_h
        kge_tails = repr_t
        attentions = self.co_attention(kge_heads, kge_tails)
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        return scores     

#intra+inter
class MVN_DDI_Block(nn.Module):
    def __init__(self, in_features, n_heads, head_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = GATConv(in_features, head_out_feats, n_heads, edge_dim=64)
        self.intraAtt = IntraGraphAttention(head_out_feats*n_heads)
        self.interAtt = InterGraphAttention(head_out_feats*n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
        self.norm = LayerNorm(n_heads * head_out_feats)

        self.lin_up = Linear(64, 64, bias=True, weight_initializer='glorot')

        self.re_shape_e = Linear(64, 128, bias=True, weight_initializer='glorot')
        
    
    def forward(self, h_data,t_data,b_graph,h_data_edge, t_data_edge):
     
        h_data.x = self.feature_conv(h_data.x, h_data.edge_index, h_data.edge_attr)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index, t_data.edge_attr)
        h_data.edge_attr = self.lin_up(h_data.edge_attr)
        t_data.edge_attr = self.lin_up(t_data.edge_attr)
   
        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)
        
        h_interRep,t_interRep = self.interAtt(h_data,t_data,b_graph)
        
        h_rep = torch.cat([h_intraRep,h_interRep],1)
        t_rep = torch.cat([t_intraRep,t_interRep],1)
        h_data.x = h_rep
        t_data.x = t_rep

        
        # readout
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores= self.readout(h_data.x, h_data.edge_index, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores= self.readout(t_data.x, t_data.edge_index, batch=t_data.batch)
      
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)

        h_data_edge.x = h_data.edge_attr
        t_data_edge.x = t_data.edge_attr
        h_global_graph_emb_edge = global_add_pool(h_data_edge.x, batch=h_data_edge.batch)
        t_global_graph_emb_edge = global_add_pool(t_data_edge.x, batch=t_data_edge.batch)
        h_global_graph_emb_edge = F.relu(self.re_shape_e(h_global_graph_emb_edge))
        t_global_graph_emb_edge = F.relu(self.re_shape_e(t_global_graph_emb_edge))

        h_global_graph_emb = h_global_graph_emb + h_global_graph_emb_edge
        t_global_graph_emb = t_global_graph_emb + t_global_graph_emb_edge


        h_data.x = F.relu(self.norm(h_data.x, h_data.batch))
        t_data.x = F.relu(self.norm(t_data.x, t_data.batch))
        h_data.edge_attr = F.relu(h_data.edge_attr)
        t_data.edge_attr = F.relu(t_data.edge_attr)
        
        return h_data,t_data, h_global_graph_emb,t_global_graph_emb




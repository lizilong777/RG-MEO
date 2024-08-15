import torch
import torch.nn as nn
from .Model import Model
from utils import generate_sampled_graph_and_labels, uniform
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch.nn.functional import leaky_relu

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


class RGCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))
        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)

# class RGCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, num_relations, num_bases,
#                  root_weight=True, bias=True, **kwargs):
#         super(RGCNConv, self).__init__(aggr='add', **kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_relations = num_relations
#         self.num_bases = num_bases
#         self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
#         self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))
#         self.att_weight = nn.Parameter(torch.Tensor(2 * out_channels, 1))
#         self.linear_W = nn.Linear(in_channels, out_channels, bias=True)
#         # if root_weight:
#         #     self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
#         # else:
#         #     self.register_parameter('root', None)
#         # if bias:
#         #     self.bias = nn.Parameter(torch.Tensor(out_channels))
#         # else:
#         #     self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         size = self.num_bases * self.in_channels
#         nn.init.xavier_uniform_(self.basis)
#         nn.init.xavier_uniform_(self.att)
#         nn.init.xavier_uniform_(self.att_weight)
#         nn.init.xavier_uniform_(self.linear_W.weight)
#         # if self.root is not None:
#         #     nn.init.xavier_uniform_(self.root)
#         # if self.bias is not None:
#         #     nn.init.zeros_(self.bias)
#
#     def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
#         # x= [14388,400]
#         return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_norm=edge_norm)
#
#     def message(self, x_j, x_i, edge_index_j, edge_type, edge_norm):
#         '''
#         MKG-Y
#         out shape: torch.Size([21310, 400])
#         e_ij shape: torch.Size([21310])
#         edge_index_j shape: torch.Size([21310])
#         alpha shape: torch.Size([21310])
#         这个代码应该是为每一条边分配一个注意力权重了w
#         '''
#         w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
#         w = w.view(self.num_relations, self.in_channels, self.out_channels)
#         w = torch.index_select(w, 0, edge_type)
#         out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
#         if edge_norm is not None:
#             out = out * edge_norm.view(-1, 1)
#
#         x_i_transformed = self.linear_W(x_i)
#         out_transformed = self.linear_W(out)
#         cat_x = torch.cat([x_i_transformed, out_transformed], dim=-1)
#         e_ij = F.leaky_relu(torch.matmul(cat_x, self.att_weight).squeeze(-1))
#         alpha = softmax(e_ij, edge_index_j)
#         out = out_transformed * alpha.view(-1, 1)
#
#         return out
#
#     def update(self, aggr_out, x):
#         # if self.root is not None:
#         #     if x is None:
#         #         out = aggr_out + self.root
#         #     else:
#         #         out = aggr_out + torch.matmul(x, self.root)
#         out = aggr_out
#         # if self.bias is not None:
#         #     out = out + self.bias
#         return out


class AdvMixRotatE(Model):
    def __init__(
        self,
        ent_tot,
        rel_tot,
        dim=100,
        margin=6.0,
        epsilon=2.0,
        img_emb=None,
        text_emb=None,
    ):
        super(AdvMixRotatE, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None
        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim * 2
        self.dim_r = dim
        # 初始化一个矩阵形状[ent_tot,dim_e]
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )
        self.img_dim = img_emb.shape[1]
        self.text_dim = text_emb.shape[1]
        self.img_proj = nn.Linear(self.img_dim, self.dim_e)
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(True)
        self.text_proj = nn.Linear(self.text_dim, self.dim_e)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)
        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
        self.ent_attn.requires_grad_(True)
        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        # new
        # 初始化RGCN
        self.conv1 = RGCNConv(
            self.dim_e, self.dim_e, self.rel_tot, num_bases=4)
        self.conv2 = RGCNConv(
            self.dim_e, self.dim_e, self.rel_tot, num_bases=4)
        self.dropout_ratio = 0.2
        # 关系感知因子
        self.network_t = nn.Sequential(
            nn.Linear(self.dim_r, self.dim_e, bias=True),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e, bias=True),
            nn.Sigmoid()
        )
        self.network_v = nn.Sequential(
            nn.Linear(self.dim_r, self.dim_e, bias=True),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e, bias=True),
            nn.Sigmoid()
        )


    def RGCN_forward(self, entity, edge_index, edge_type, edge_norm):
        x = self.ent_embeddings(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, edge_type, edge_norm)
        return x

    def relation_factor_mlt(self, text_emb, img_emb, rel_emb):
        if rel_emb.size(0) != text_emb.size(0):
            rel_emb = rel_emb.expand(text_emb.size(0), -1)

        a_r_t = self.network_t(rel_emb)
        adjust_text_emb = text_emb * a_r_t

        a_r_v = self.network_t(rel_emb)
        adjust_img_emb = img_emb * a_r_v

        return adjust_text_emb, adjust_img_emb


    def get_joint_embeddings(self, es, ev, et):
        e = torch.stack((es, ev, et), dim=1)
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=-1)
        print("attention_weights", attention_weights[0])
        context_vectors = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)
        return context_vectors

    # RotatE计算score
    def _calc(self, h, t, r, mode):
        pi = self.pi_const

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == "head_batch":
            re_score = (re_relation * re_tail + im_relation * im_tail) - re_head
            im_score = (re_relation * im_tail - im_relation * re_tail) - im_head
        else:
            re_score = (re_head * re_relation - im_head * im_relation) - re_tail
            im_score = (re_head * im_relation + im_head * re_relation) - im_tail

        score = torch.stack([re_score, im_score], dim=0).norm(dim=0).sum(dim=-1)
        return score.flatten()

    def forward(self, data):
        # 测试：头实体个数为15000个，尾实体个数为1个
        # batch_h.shape = [132096]
        mode = data['mode']
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        # device = batch_r.device
        # train_data = generate_sampled_graph_and_labels(self.rel_tot, device)
        # ent_emb = self.RGCN_forward(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
        # self.ent_embeddings.weight.data[train_data.entity] = ent_emb
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        # new
        # h_text_emb_adjust, h_img_emb_adjust = self.relation_factor_mlt(h_text_emb, h_img_emb, r)
        # h_joint = self.get_joint_embeddings(h, h_img_emb_adjust, h_text_emb_adjust)
        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb)
        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb)
        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score
    
    def get_batch_ent_embs(self, data):
        return self.ent_embeddings(data)

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul
    
    def get_attention(self, es, ev, et):
        # es, ev, et: [num_ent, emb_dim]
        e = torch.stack((es, ev, et), dim=1)
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=-1)
        return attention_weights

    def get_attention_weight(self, h, t):
        h = torch.LongTensor([h])
        t = torch.LongTensor([t])
        h_s = self.ent_embeddings(h)
        t_s = self.ent_embeddings(t)
        h_img_emb = self.img_proj(self.img_embeddings(h))
        t_img_emb = self.img_proj(self.img_embeddings(t))
        h_text_emb = self.text_proj(self.text_embeddings(h))
        t_text_emb = self.text_proj(self.text_embeddings(t))
        # the fake joint embedding
        h_attn = self.get_attention(h_s, h_img_emb, h_text_emb)
        t_attn = self.get_attention(t_s, t_img_emb, t_text_emb)
        return h_attn, t_attn

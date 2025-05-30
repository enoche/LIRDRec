import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct

from common.abstract_recommender import GeneralRecommender


class LIRDRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LIRDRec, self).__init__(config, dataset)
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']
        self.decay_base = config['decay_base']
        self.decay_weight = config['decay_weight']
        self.cur_epoch = 0

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path,
                                   'mm_adj_lirdrec_{}_{}.pt'.format(self.knn_k, int(10 * self.mm_image_weight)))

        v_feat_dim, t_feat_dim = 0, 0
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            v_feat_dim = self.v_feat.shape[1]

            self.v_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(self.n_users, self.embedding_dim), dtype=torch.float32, requires_grad=True), gain=1).to(
                self.device))
            self.v_MLP = nn.Linear(v_feat_dim, 4 * self.embedding_dim)
            self.v_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            t_feat_dim = self.t_feat.shape[1]
            self.t_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(self.n_users, self.embedding_dim), dtype=torch.float32, requires_grad=True), gain=1).to(
                self.device))
            self.t_MLP = nn.Linear(t_feat_dim, 4 * self.embedding_dim)
            self.t_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)

        self.id_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(self.n_users, self.embedding_dim), dtype=torch.float32, requires_grad=True), gain=1).to(
            self.device))
        self.s_MLP = nn.Linear(t_feat_dim + v_feat_dim, 4 * self.embedding_dim)
        self.s_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)
        self.fusion_module = PWC(self.n_users, self.embedding_dim, self.embedding_dim // 4, self.device,
                                          self.decay_base, self.decay_weight)

        w_t = dct.dct(self.t_feat, norm='ortho')
        w_v = dct.dct(self.v_feat, norm='ortho')
        self.interleaved_feat = torch.cat((w_v, w_t), 1)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        self.result_embed = None

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        self.cur_epoch += 1
        self.fusion_module.update_w(self.cur_epoch)

        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):
        # mm feature via gcns
        tmp_v_feat = self.v_MLP_1(F.leaky_relu(self.v_MLP(self.v_feat)))
        tmp_t_feat = self.t_MLP_1(F.leaky_relu(self.t_MLP(self.t_feat)))
        tmp_s_features = self.interleaved_feat
        tmp_s_feat = self.s_MLP_1(F.leaky_relu(self.s_MLP(tmp_s_features)))

        rep_uv = torch.cat((self.v_preference, tmp_v_feat), dim=0)
        rep_ut = torch.cat((self.t_preference, tmp_t_feat), dim=0)
        rep_sh = torch.cat((self.id_preference, tmp_s_feat), dim=0)
        v_x = torch.cat((F.normalize(rep_uv), F.normalize(rep_ut), F.normalize(rep_sh)), 1)

        # gcns
        ego_emb = v_x
        all_embeddings = [ego_emb]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_emb)
            ego_emb = side_embeddings
            all_embeddings += [ego_emb]
        representation = torch.stack(all_embeddings, dim=0).sum(dim=0)

        # User/item
        item_rep = representation[self.n_users:]

        vt_rep = representation[:self.n_users]
        v_rep = vt_rep[:, :self.embedding_dim]
        t_rep = vt_rep[:, self.embedding_dim: 2 * self.embedding_dim]
        s_rep = vt_rep[:, 2 * self.embedding_dim:]

        user_rep = self.fusion_module(v_rep, t_rep, s_rep)

        # item gcns
        h = item_rep
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        item_rep = item_rep + h
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        return user_rep, item_rep

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        reg_u_loss = (ua_embeddings ** 2).mean()
        reg_i_loss = (ia_embeddings ** 2).mean()

        reg_loss = self.reg_weight * (reg_u_loss + reg_i_loss)

        return batch_mf_loss + reg_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix


class WeCopy(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(WeCopy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        a = self.fc2(h)
        return a


class PWC(nn.Module):
    def __init__(self, n_users, input_dim, hidden_dim, device, base, w1):
        super(PWC, self).__init__()
        self.weco_a = WeCopy(input_dim, hidden_dim)
        self.weco_b = WeCopy(input_dim, hidden_dim)
        self.weco_c = WeCopy(input_dim, hidden_dim)
        self.theta = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(n_users, 3), dtype=torch.float32, requires_grad=True).to(device)))
        self.w1, self.w2 = w1, 1 - w1
        self.base = base
        self.last_att = self.theta.data

    def update_w(self, epoch):
        self.w1, self.w2 = update_weights(self.w1, self.w2, epoch, base=self.base)
        self.theta.data = self.last_att

    def forward(self, a, b, c):
        att_a = self.weco_a(a)
        att_b = self.weco_b(b)
        att_c = self.weco_c(c)
        f_att = torch.cat((att_a, att_b, att_c), dim=1)
        # fusion
        _att2 = self.w1 * f_att + self.w2 * self.theta
        _att2 = F.softmax(_att2, dim=1)
        self.last_att = _att2

        _att2 = torch.unsqueeze(_att2, dim=1)
        fused_representation = torch.cat((_att2[:, :, 0] * a, _att2[:, :, 1] * b, _att2[:, :, 2] * c), dim=1)
        return fused_representation


import math

def update_weights(w1, w2, epoch, base=0.9):
    """
    Update the weights exponentially over training epochs.

    Parameters:
    w1 (float): Initial weight for the first network.
    w2 (float): Initial weight for the second network.
    epoch (int): Current epoch number.
    total_epochs (int): Total number of training epochs.
    base (float): Base of the exponential, determines the rate of weight update.
    Smaller base will result in a faster transfer of weight from the first network to the second.

    Returns:
    tuple: Updated weights (w1, w2).
    """
    # Calculate the decay factor for the current epoch
    decay_factor = math.pow(base, epoch)

    # Update w1 and w2 based on the decay factor
    w1_updated = w1 * decay_factor
    w2_updated = w2

    # Normalize the weights to ensure they sum up to 1
    weight_sum = w1_updated + w2_updated
    w1_normalized = w1_updated / weight_sum
    w2_normalized = w2_updated / weight_sum

    return w1_normalized, w2_normalized

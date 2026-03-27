"""
Model definitions for the FCN-SC framework.

Modules
-------
- supervised_contrastive_loss   : SupCon loss (Khosla et al., 2020)
- GroupFeatureEncoder           : Dataset-aware group feature encoder
- DualPrototypeCouplingModule   : Dual-prototype coupling + auxiliary losses
- FCN_SC_v5                     : Full network combining the above
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

from utils import triu_flatten


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------

def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor,
                                 tau: float = 0.07) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    Args:
        z      : (N, D) L2-normalised representation vectors.
        labels : (N,)  Class labels (0 or 1).
        tau    : Temperature scalar.

    Returns:
        Scalar loss averaged over all anchor nodes that have at least one
        positive counterpart.
    """
    n      = z.size(0)
    device = z.device

    sim = torch.matmul(z, z.T) / tau
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    pos_mask  = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~self_mask

    exp_sim = torch.exp(sim)
    denom   = (exp_sim * ~self_mask).sum(dim=1, keepdim=True)
    log_prob = sim - torch.log(denom + 1e-8)

    pos_count = pos_mask.float().sum(dim=1)
    valid     = pos_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    loss_per = -(log_prob * pos_mask.float()).sum(dim=1) / (pos_count + 1e-8)
    return loss_per[valid].mean()


# ---------------------------------------------------------------------------
# Group Feature Encoder
# ---------------------------------------------------------------------------

class GroupFeatureEncoder(nn.Module):
    """
    Encodes heterogeneous node features into a set of group representations.

    Feature layout per dataset
    --------------------------
    twibot-20  : [num(5) | tweets(768) | cat(3) | des(768)]
    Cresci-15  : [cat(1) | num(5) | des(768) | tweets(768)]
    MGTAB      : [mixed(20) | des(768)]  — cat idx=[0,1,2,4,8,15-19], num idx rest
    """

    # Fixed feature indices for MGTAB
    _MGTAB_CAT_IDX = torch.tensor([0, 1, 2, 4, 8, 15, 16, 17, 18, 19])
    _MGTAB_NUM_IDX = torch.tensor([3, 5, 6, 7, 9, 10, 11, 12, 13, 14])

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.dataset       = args.dataset
        self.num_relations = args.num_relations
        d                  = args.group_dim

        if args.dataset in ("twibot-20", "Cresci-15"):
            cat_dim = 3 if args.dataset == "twibot-20" else 1
            self.enc_des = nn.Sequential(
                nn.Linear(768, d * 2), nn.PReLU(), nn.Linear(d * 2, d))
            self.enc_twt = nn.Sequential(
                nn.Linear(768, d * 2), nn.PReLU(), nn.Linear(d * 2, d))
            self.enc_cat = nn.Sequential(
                nn.Linear(cat_dim, d), nn.PReLU(), nn.Linear(d, d))
            self.enc_num = nn.Sequential(
                nn.Linear(5, d), nn.PReLU(), nn.Linear(d, d))
        else:  # MGTAB
            self.register_buffer("cat_idx", self._MGTAB_CAT_IDX)
            self.register_buffer("num_idx", self._MGTAB_NUM_IDX)
            self.enc_des = nn.Sequential(
                nn.Linear(768, d * 2), nn.PReLU(), nn.Linear(d * 2, d))
            self.enc_cat = nn.Sequential(
                nn.Linear(10, d), nn.PReLU(), nn.Linear(d, d))
            self.enc_num = nn.Sequential(
                nn.Linear(10, d), nn.PReLU(), nn.Linear(d, d))

        self.rel_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(2, d), nn.PReLU(), nn.Linear(d, d))
            for _ in range(args.num_relations)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Args:
            x          : (N, F) Node feature matrix.
            edge_index : (2, E) Edge index tensor.
            edge_type  : (E,)   Edge relation type.
            num_nodes  : Total number of nodes N.

        Returns:
            group_reprs: (N, K, d)  K group representations per node.
        """
        device = x.device
        groups = []

        if self.dataset == "twibot-20":
            groups.append(self.enc_des(x[:, -768:]))
            groups.append(self.enc_twt(x[:, 5:5 + 768]))
            groups.append(self.enc_cat(x[:, 5 + 768:5 + 768 + 3]))
            groups.append(self.enc_num(x[:, :5]))
        elif self.dataset == "Cresci-15":
            groups.append(self.enc_des(x[:, 6:774]))
            groups.append(self.enc_twt(x[:, 774:1542]))
            groups.append(self.enc_cat(x[:, 0:1]))
            groups.append(self.enc_num(x[:, 1:6]))
        else:  # MGTAB
            groups.append(self.enc_des(x[:, -768:]))
            groups.append(self.enc_cat(x[:, self.cat_idx]))
            groups.append(self.enc_num(x[:, self.num_idx]))

        row, col = edge_index
        for r in range(self.num_relations):
            mask = edge_type == r
            in_deg = torch.log1p(
                degree(col[mask], num_nodes, dtype=torch.float)
                if mask.sum() > 0 else torch.zeros(num_nodes, device=device)
            ).unsqueeze(1)
            out_deg = torch.log1p(
                degree(row[mask], num_nodes, dtype=torch.float)
                if mask.sum() > 0 else torch.zeros(num_nodes, device=device)
            ).unsqueeze(1)
            groups.append(self.rel_encoders[r](torch.cat([in_deg, out_deg], dim=1)))

        return torch.stack(groups, dim=1)  # (N, K, d)


# ---------------------------------------------------------------------------
# Dual Prototype Coupling Module
# ---------------------------------------------------------------------------

class DualPrototypeCouplingModule(nn.Module):
    """
    Maintains human/bot prototype coupling matrices and computes:
      - coupling features for classification
      - prototype alignment losses
      - prototype separation losses
      - supervised contrastive loss on coupling representations
    """

    def __init__(self, num_groups: int):
        super().__init__()
        K        = num_groups
        self.K   = K
        triu_dim = K * (K - 1) // 2

        self.M_human = nn.Parameter(torch.zeros(K, K))
        self.M_bot   = nn.Parameter(torch.zeros(K, K))
        self.W_h_raw = nn.Parameter(torch.ones(K, K))
        self.W_b_raw = nn.Parameter(torch.ones(K, K))

        self.proj_head = nn.Sequential(
            nn.Linear(triu_dim, triu_dim * 2),
            nn.ReLU(),
            nn.Linear(triu_dim * 2, triu_dim),
        )

    @property
    def W_h(self) -> torch.Tensor:
        return F.softplus(self.W_h_raw)

    @property
    def W_b(self) -> torch.Tensor:
        return F.softplus(self.W_b_raw)

    def compute_coupling_matrix(self, group_reprs: torch.Tensor) -> torch.Tensor:
        """Compute normalised group-pair cosine similarity matrix. Returns (N, K, K)."""
        g_norm = F.normalize(group_reprs, dim=2)
        return torch.bmm(g_norm, g_norm.transpose(1, 2))

    def frob_dist_triu(self, C_batch: torch.Tensor,
                       M: torch.Tensor) -> torch.Tensor:
        """Frobenius distance on upper-triangular elements. Returns (B,)."""
        return triu_flatten(C_batch - M, self.K).pow(2).sum(dim=1).sqrt()

    def compute_features(self, C: torch.Tensor):
        """
        Compute coupling features from coupling matrix C (N, K, K).

        Returns
        -------
        R_h_triu  : (N, triu_dim)
        R_b_triu  : (N, triu_dim)
        dist_h    : (N, 1)
        dist_b    : (N, 1)
        dist_diff : (N, 1)
        """
        K   = self.K
        M_h = self.M_human.unsqueeze(0)
        M_b = self.M_bot.unsqueeze(0)

        R_h_triu  = triu_flatten(self.W_h.unsqueeze(0) * (C - M_h), K)
        R_b_triu  = triu_flatten(self.W_b.unsqueeze(0) * (C - M_b), K)
        dist_h    = self.frob_dist_triu(C, M_h).unsqueeze(1)
        dist_b    = self.frob_dist_triu(C, M_b).unsqueeze(1)
        dist_diff = dist_b - dist_h
        return R_h_triu, R_b_triu, dist_h, dist_b, dist_diff

    def compute_aux_losses(self, C: torch.Tensor, train_idx, labels: torch.Tensor,
                           margin: float):
        """
        Prototype alignment and separation losses (identical to v14).

        Returns
        -------
        loss_proto_h, loss_proto_b, loss_sep_h, loss_sep_b : scalars
        """
        device      = C.device
        train_idx_t = torch.tensor(train_idx, device=device)
        y_tr        = labels[train_idx_t]
        zero        = torch.tensor(0.0, device=device)

        human_mask = y_tr == 0
        bot_mask   = y_tr == 1
        if human_mask.sum() < 2 or bot_mask.sum() < 2:
            return zero, zero, zero, zero

        C_human = C[train_idx_t[human_mask]]
        C_bot   = C[train_idx_t[bot_mask]]
        M_h     = self.M_human.unsqueeze(0)
        M_b     = self.M_bot.unsqueeze(0)

        loss_proto_h = (self.M_human - C_human.mean(0).detach()).pow(2).mean()
        loss_proto_b = (self.M_bot   - C_bot.mean(0).detach()).pow(2).mean()

        d_h2Mh = self.frob_dist_triu(C_human, M_h)
        d_b2Mh = self.frob_dist_triu(C_bot,   M_h)
        d_h2Mb = self.frob_dist_triu(C_human, M_b)
        d_b2Mb = self.frob_dist_triu(C_bot,   M_b)

        loss_sep_h = F.relu(margin - (d_b2Mh.mean() - d_h2Mh.mean()))
        loss_sep_b = F.relu(margin - (d_h2Mb.mean() - d_b2Mb.mean()))

        return loss_proto_h, loss_proto_b, loss_sep_h, loss_sep_b

    def compute_supcon_loss(self, C: torch.Tensor, train_idx,
                            labels: torch.Tensor, n_sup: int,
                            tau: float) -> torch.Tensor:
        """
        Supervised Contrastive Loss computed in the triu coupling space.

        Representation: z = L2_norm(triu_flatten(C[sampled_train_nodes]))
        Gradient path : l_sc → z → triu(C) → C → GroupFeatureEncoder

        Sampling: draw n_sup nodes from the training set each epoch,
        ensuring both classes are represented (returns 0 otherwise).
        """
        device      = C.device
        train_idx_t = torch.tensor(train_idx, device=device)
        y_tr        = labels[train_idx_t]

        valid       = y_tr >= 0
        train_idx_t = train_idx_t[valid]
        y_tr        = y_tr[valid]

        n_train = train_idx_t.size(0)
        if n_train < 4:
            return torch.tensor(0.0, device=device)

        n_sample = min(n_sup, n_train)
        perm     = torch.randperm(n_train, device=device)[:n_sample]
        idx_sub  = train_idx_t[perm]
        y_sub    = y_tr[perm]

        if y_sub.unique().size(0) < 2:
            return torch.tensor(0.0, device=device)

        z = triu_flatten(C[idx_sub], self.K)
        z = F.normalize(z, dim=1)
        return supervised_contrastive_loss(z, y_sub, tau)


# ---------------------------------------------------------------------------
# Full Network
# ---------------------------------------------------------------------------

class FCN_SC_v5(nn.Module):
    """
    Feature Coupling Network with Supervised Contrastive loss (v5).

    Architecture
    ------------
    GroupFeatureEncoder → DualPrototypeCouplingModule → MLP Classifier

    During training, four auxiliary losses are returned alongside logits:
    lph, lpb (prototype alignment), lsh, lsb (prototype separation), l_sc (SupCon).
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args          = args
        self.num_relations = args.num_relations
        hidden             = args.hidden_dim

        self.K = (4 if args.dataset in ("twibot-20", "Cresci-15") else 3) + args.num_relations

        triu_dim  = self.K * (self.K - 1) // 2
        clf_input = 2 * triu_dim

        self.group_encoder   = GroupFeatureEncoder(args)
        self.coupling_module = DualPrototypeCouplingModule(self.K)
        self.classifier      = nn.Sequential(
            nn.Linear(clf_input, hidden * 2),
            nn.PReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden * 2, hidden),
            nn.PReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, data, train_idx=None, labels=None):
        """
        Args:
            data      : PyG Data object.
            train_idx : Training node indices (required during training).
            labels    : Node labels (required during training).

        Returns:
            logits : (N, 2)
            lph, lpb, lsh, lsb, l_sc : scalar auxiliary losses (0 during eval).
        """
        x         = data.x
        num_nodes = data.num_total

        group_reprs = self.group_encoder(
            x, data.edge_index, data.edge_type, num_nodes
        )
        C = self.coupling_module.compute_coupling_matrix(group_reprs)

        R_h_triu, R_b_triu, dist_h, dist_b, dist_diff = \
            self.coupling_module.compute_features(C)

        logits = self.classifier(torch.cat([R_h_triu, R_b_triu], dim=1))

        zero = torch.tensor(0.0, device=x.device)
        lph = lpb = lsh = lsb = l_sc = zero

        if self.training and train_idx is not None and labels is not None:
            lph, lpb, lsh, lsb = self.coupling_module.compute_aux_losses(
                C, train_idx, labels, self.args.margin
            )
            l_sc = self.coupling_module.compute_supcon_loss(
                C, train_idx, labels, self.args.n_sup, self.args.tau
            )

        return logits, lph, lpb, lsh, lsb, l_sc

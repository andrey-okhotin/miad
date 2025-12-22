import torch
import torch.nn as nn

import math




class CSPNet_SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space
 
    def forward(self, x):
        shape = x.shape[:-1]
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.reshape(*shape, self.dim)




class CSPNetLight_MessagePassing(nn.Module):
    def __init__(self, hidden_dim, frac_freq):
        super(CSPNetLight_MessagePassing, self).__init__()
        self.lat_embed_dim = 9
        self.frac_embed_dim = 6 * frac_freq
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.lat_embed_dim + self.frac_embed_dim, 1 * hidden_dim),
            nn.SiLU(),
            nn.Linear(1 * hidden_dim, 1 * hidden_dim),
            nn.SiLU()
        )

    def forward(self, node_embed, edge_embed, graph_embed):
        bs, N_m, h_dim = node_embed.shape
        l_dim, f_dim = self.lat_embed_dim, self.frac_embed_dim

        fc_graph_edges = torch.cat((
            node_embed[:,None,:,:].repeat(1,N_m,1,1).reshape(bs,N_m*N_m,h_dim),
            node_embed[:,:,None,:].repeat(1,1,N_m,1).reshape(bs,N_m*N_m,h_dim),
            graph_embed[:,None,None,:].repeat(1,N_m,N_m,1).reshape(bs,N_m*N_m,l_dim),
            edge_embed.reshape(bs,N_m*N_m,f_dim)
        ), dim=-1).reshape(bs,N_m,N_m,2*h_dim+l_dim+f_dim)
        
        edge_msg = self.mlp(fc_graph_edges)
        node_msg = edge_msg.mean(dim=2)
        return node_msg




class CSPNetLight_MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(CSPNetLight_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 1 * hidden_dim),
            nn.SiLU(),
            nn.Linear(1 * hidden_dim, 1 * hidden_dim),
            nn.SiLU()
        )
    
    def forward(self, node_embed, node_msg):
        h = self.mlp(torch.cat((
            node_embed, 
            node_msg
        ), dim=-1))
        return h




class CSPNetLight_Block(nn.Module):
    def __init__(self, hidden_dim, frac_freq):
        super(CSPNetLight_Block, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.message_passing = CSPNetLight_MessagePassing(hidden_dim, frac_freq)
        self.mlp = CSPNetLight_MLP(hidden_dim)
    
    def forward(self, node_embed, edge_embed, graph_embed):
        normed_node_embed = self.norm(node_embed)
        node_msg = self.message_passing(normed_node_embed, edge_embed, graph_embed)
        unnormed_node_embed = normed_node_embed + self.mlp(normed_node_embed, node_msg)
        return unnormed_node_embed




class CSPNetLight(nn.Module):

    def __init__(self, 
        hidden_dim = 128,
        time_dim = 256,
        num_layers = 4,
        max_atoms = 100,
        num_freqs = 10,
        **kwargs
    ):
        super(CSPNetLight, self).__init__()
        self.N_types = max_atoms
        self.hidden_dim = hidden_dim
        self.num_blocks = num_layers
        self.frac_freq = num_freqs
        self.time_dim = time_dim
        
        # embeddings
        self.atom_embedding = nn.Linear(self.N_types, self.hidden_dim)
        self.frac_diff_encoding = CSPNet_SinusoidsEmbedding(n_frequencies=self.frac_freq, n_space=3)
        self.node_embedding = nn.Linear(hidden_dim + self.time_dim, self.hidden_dim)

        # main blocks
        self.blocks = nn.ModuleList([ 
            CSPNetLight_Block(self.hidden_dim, self.frac_freq) for _ in range(self.num_blocks) 
        ])
        self.final_layer_norm = nn.LayerNorm(self.hidden_dim)

        # heads
        self.lattice_predictor = nn.Linear(self.hidden_dim, 9)
        self.fractional_predictor = nn.Linear(self.hidden_dim, 3)
        self.atom_predictor = nn.Linear(self.hidden_dim, self.N_types, bias=False)




    def forward(
        self, 
        t,            # (bs,)
        t_embed,      # (bs, n_freq)
        atom_types,   # (bs * N_m, N_types)
        fractional,   # (bs * N_m, 3)
        lattice,      # (bs, 3, 3)
        num_atoms,    # (bs,)
        node2graph
    ):
        bs, N_m, h_dim = t.shape[0], atom_types.shape[0] // t.shape[0], self.hidden_dim

        # input
        # - lattice
        lattice_embed = (lattice @ lattice.transpose(-1,-2)).reshape(bs, 9)
        # - fractional
        f = fractional.reshape(bs, N_m, 3)                                            # (bs, N_m, 3)
        frac_diff = f[:,None,:,:] - f[:,:,None,:]
        frac_embed = self.frac_diff_encoding(frac_diff)
        # - atom types
        atom_embed = self.atom_embedding(atom_types.reshape(bs,N_m,self.N_types))     # (bs, N_m, hidden_dim)
        # - time
        t_per_atom = t_embed.repeat_interleave(num_atoms, dim=0).reshape(bs,N_m,-1)
        node_features = torch.cat([atom_embed, t_per_atom], dim=-1)
        node_embed = self.node_embedding(node_features)
        
        # body
        for block in self.blocks:
            node_embed = block(
                node_embed=node_embed, 
                edge_embed=frac_embed, 
                graph_embed=lattice_embed
            )
        node_embed = self.final_layer_norm(node_embed)
        
        # heads
        # - lattice
        lattice_pred = self.lattice_predictor(node_embed.mean(dim=1))
        lattice_mean = torch.einsum('bij,bjk->bik', lattice_pred.reshape(bs,3,3), lattice)
        # - fractional
        fractional_pred = self.fractional_predictor(node_embed)
        fractional_shift = fractional_pred.reshape(bs*N_m,3)
        # - atom types
        atom_pred = self.atom_predictor(node_embed)
        atom_type = atom_pred.reshape(bs*N_m,self.N_types) 

        return lattice_mean, fractional_shift, atom_type
    
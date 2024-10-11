import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn

class MLP(torch.nn.Module):
    def __init__(self, *sizes, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.2))
                if dropout: fcs.append(torch.nn.Dropout(p=0.2))
                if batchnorm: fcs.append(torch.nn.BatchNorm1d(sizes[i]))
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x): 
        return self.layers(x) 
    

class TimingGNN(torch.nn.Module):
    def __init__(self, in_nf, in_ef, out_nf, h1=32, h2=32):
        super(TimingGNN, self).__init__()
        self.in_nf = in_nf  # node feature dimension
        self.in_ef = in_ef  # edge feature dimension
        self.out_nf = out_nf  # output node feature dimension
        self.h1 = h1  # hidden layer for sum and max
        self.h2 = h2

        # MLP for cellarc
        self.MLP_msg_forward_cellarc = MLP(self.in_nf * 2 + 4, 64, 64, self.h1 + self.h2 + 1)  # cellarc forward propagation
        self.MLP_msg_backward_cellarc = MLP(self.in_nf * 2 + 6, 64, 64, self.h1 + self.h2)  # cellarc backward propagation

        # MLP for netarc
        self.MLP_msg_forward_netarc = MLP(self.in_nf * 2 + self.in_ef, 64, 64, self.h1 + self.h2)  # netarc forward propagation
        self.MLP_msg_backward_netarc = MLP(self.in_nf * 2 + self.in_ef, 64, 64, self.h1 + self.h2)  # netarc backward propagation

        # MLP for node feature aggregation
        self.MLP_reduce = MLP(self.in_nf + 4 * self.h1 + 4 * self.h2, 64, 64, self.out_nf)

    # cellarc forward propagation
    def message_func_forward_cellarc(self, edges):
        src_features = edges.src['nf']
        dst_features = edges.dst['nf']
        edge_features = edges.data['feature'][:, -4:]
        combined_features = torch.cat([src_features, dst_features, edge_features], dim=1)
        x = self.MLP_msg_forward_cellarc(combined_features)
        k, f1, f2 = torch.split(x, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)
        return {'msg_f1': f1 * k, 'msg_f2': f2 * k}

    # cellarc backward propagation
    def message_func_backward_cellarc(self, edges):
        src_features = edges.src['nf']
        dst_features = edges.dst['nf']
        edge_features = edges.data['feature'][:, :6]
        combined_features = torch.cat([src_features, dst_features, edge_features], dim=1)
        x = self.MLP_msg_backward_cellarc(combined_features)
        return {'msg_b': x}

    # netarc forward/backward propagation
    def message_func_netarc(self, edges, is_forward=True):
        src_features = edges.src['nf']
        dst_features = edges.dst['nf']
        edge_features = edges.data['feature']  # edge feature
        combined_features = torch.cat([src_features, dst_features, edge_features], dim=1)
        x = self.MLP_msg_forward_netarc(combined_features) if is_forward else self.MLP_msg_backward_netarc(combined_features)
        return {'msg_n': x}

    # Node feature aggregation (for cellarc and netarc)
    def node_reduce_o(self, nodes):
        aggregated_f1 = nodes.data['n_f1_sum'] if 'n_f1_sum' in nodes.data else torch.zeros((nodes.batch_size(), self.h1), device=nodes.device)
        aggregated_f2 = nodes.data['n_f2_max'] if 'n_f2_max' in nodes.data else torch.zeros((nodes.batch_size(), self.h2), device=nodes.device)

        aggregated_msg_n = nodes.data['n_n_mean'] if 'n_n_mean' in nodes.data else torch.zeros((nodes.batch_size(), self.h1 + self.h2), device=nodes.device)

        aggregated_msg_n_back = nodes.data['n_n_back_mean'] if 'n_n_back_mean' in nodes.data else torch.zeros((nodes.batch_size(), self.h1 + self.h2), device=nodes.device)

        aggregated_msg_b = nodes.data['n_b_mean'] if 'n_b_mean' in nodes.data else torch.zeros((nodes.batch_size(), self.h1 + self.h2), device=nodes.device)

        concatenated_messages = torch.cat([nodes.data['nf'], aggregated_f1, aggregated_f2, aggregated_msg_b, aggregated_msg_n, aggregated_msg_n_back], dim=1)

        new_nf = self.MLP_reduce(concatenated_messages)
        return {'new_nf': new_nf}

    def forward(self, g, nf):
        with g.local_scope():
            g.ndata['nf'] = nf

            # Cellarc forward propagation
            g.update_all(self.message_func_forward_cellarc, fn.sum('msg_f1', 'n_f1_sum'), etype='cellarc')
            g.update_all(self.message_func_forward_cellarc, fn.max('msg_f2', 'n_f2_max'), etype='cellarc')

            # Cellarc backward propagation
            g.update_all(self.message_func_backward_cellarc, fn.mean('msg_b', 'n_b_mean'), etype='cellarc')

            # Netarc forward propagation (average msg_n)
            g.update_all(lambda edges: self.message_func_netarc(edges, is_forward=True), fn.mean('msg_n', 'n_n_mean'), etype='netarc')

            # Netarc backward propagation
            g.update_all(lambda edges: self.message_func_netarc(edges, is_forward=False), fn.mean('msg_n', 'n_n_back_mean'), etype='netarc')

            g.apply_nodes(self.node_reduce_o)

            return g.ndata['new_nf']


class MultiLayerTimingGNN(torch.nn.Module):
    def __init__(self, num_layers, hidden_nf, out_nf, in_nf = 5, in_ef = 4, h1=32, h2=32):
        super(MultiLayerTimingGNN, self).__init__()
        self.num_layers = num_layers
        self.gnn_layers = torch.nn.ModuleList()

        # First layer
        self.gnn_layers.append(TimingGNN(in_nf, in_ef, hidden_nf, h1=h1, h2=h2))

        # Middle layers
        for _ in range(1, num_layers - 1):
            self.gnn_layers.append(TimingGNN(hidden_nf, in_ef, hidden_nf, h1=h1, h2=h2))

        # Last layer
        self.gnn_layers.append(TimingGNN(hidden_nf, in_ef, out_nf, h1=h1, h2=h2))

    def forward(self, g):
        nf = g.ndata['feature']

        # First layer propagation using initial node features
        nf = self.gnn_layers[0](g, nf)

        # Propagate through remaining layers
        for i in range(1, self.num_layers):
            nf = self.gnn_layers[i](g, nf)

        return nf

class CNN(torch.nn.Module):
    def __init__(self, in_channels=4):
        super(CNN, self).__init__()
        
        # input 4*512*512
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2, padding=1)  # output 32*256*256
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)  # output 64*128*128
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)  # output 32*64*64
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)   # output 1*64*64

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x
    
class MultiModalNN(torch.nn.Module):
    def __init__(self, num_layers, hidden_nf, out_nf, output, in_nf = 5, in_ef = 4, h1=32, h2=32, in_channels=4):
        super(MultiModalNN, self).__init__()
        self.gnn = MultiLayerTimingGNN(num_layers, hidden_nf, out_nf, in_nf, in_ef, h1, h2)
        self.cnn = CNN(in_channels)
        self.MLP_cnn_forward = MLP(64*64, 64, 64, 32)   # MLP after Padding
        self.MLP_gnn_forward = MLP(out_nf, 64, 64, 32) # MLP after pooling
        self.MLP_CPath_Gate = MLP(20, 64, 64, 32) # MLP for Gate feature on CPath
        self.MLP_Output = MLP(96, 64, 64, output)
        
    def forward(self, g, img, padding_mask, node_num, Gate_num):
        edge_ids = g.edge_ids(Gate_num[0], Gate_num[1], etype='cellarc')
        Gate_ef = g.edata['feature'][('node', 'cellarc', 'node')][edge_ids].view(-1)
        Gate_nf = g.ndata['CPath'][Gate_num].view(-1)
        if torch.isnan(Gate_nf).any():
            raise ValueError(f"The tensor contains NaN values! Gate not in Critical Path.")
        Gate_feature = torch.cat([Gate_nf, Gate_ef], dim=0)
        nf = self.gnn(g)
        nf = nf[node_num]
        pooled_nf = torch.mean(nf, 0)
        img = self.cnn(img).squeeze(0)
        img = torch.mul(img, padding_mask).view(-1)
        Img_feature = self.MLP_cnn_forward(img)
        Gnn_feature = self.MLP_gnn_forward(pooled_nf)
        Gate_feature = self.MLP_CPath_Gate(Gate_feature)
        Output = self.MLP_Output(torch.cat([Img_feature, Gnn_feature, Gate_feature], dim=0))
        return Output
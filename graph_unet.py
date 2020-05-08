import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import copy
import time
import math
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from os.path import join as pjoin

print('using torch', torch.__version__)

# Experiment parameters
parser = argparse.ArgumentParser(description='Graph Convolutional Networks')
parser.add_argument('-D', '--dataset', type=str, default='PROTEINS')
parser.add_argument('-M', '--model', type=str, default='gcn', choices=['gcn', 'unet', 'mgcn'])
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_decay_steps', type=str, default='25,35', help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('-d', '--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('-f', '--filters', type=str, default='64,64,64', help='number of filters in each layer')
parser.add_argument('-K', '--filter_scale', type=int, default=1, help='filter scale (receptive field size), must be > 0; 1 for GCN, >1 for ChebNet')
parser.add_argument('--n_hidden', type=int, default=0,
                    help='number of hidden units in a fully connected layer after the last conv layer')
parser.add_argument('--n_hidden_edge', type=int, default=32,
                    help='number of hidden units in a fully connected layer of the edge prediction network')
parser.add_argument('--degree', action='store_true', default=False, help='use one-hot node degree features')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--bn', action='store_true', default=False, help='use BatchNorm layer')
parser.add_argument('--folds', type=int, default=10, help='number of cross-validation folds (1 for COLORS and TRIANGLES and 10 for other datasets)')
parser.add_argument('-t', '--threads', type=int, default=0, help='number of threads to load data')
parser.add_argument('--log_interval', type=int, default=10, help='interval (number of batches) of logging')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('--seed', type=int, default=111, help='random seed')
parser.add_argument('--shuffle_nodes', action='store_true', default=False, help='shuffle nodes for debugging')
parser.add_argument('-g', '--torch_geom', action='store_true', default=False, help='use PyTorch Geometric')
parser.add_argument('-a', '--adj_sq', action='store_true', default=False,
                    help='use A^2 instead of A as an adjacency matrix')
parser.add_argument('-s', '--scale_identity', action='store_true', default=False,
                    help='use 2I instead of I for self connections')
parser.add_argument('-v', '--visualize', action='store_true', default=False,
                    help='only for unet: save some adjacency matrices and other data as images')
parser.add_argument('-c', '--use_cont_node_attr', action='store_true', default=False,
                    help='use continuous node attributes in addition to discrete ones')

args = parser.parse_args()

if args.torch_geom:
    from torch_geometric.datasets import TUDataset
    import torch_geometric.transforms as T

args.filters = list(map(int, args.filters.split(',')))
args.lr_decay_steps = list(map(int, args.lr_decay_steps.split(',')))

for arg in vars(args):
    print(arg, getattr(args, arg))

n_folds = args.folds  # train,val,test splits for COLORS and TRIANGLES and 10-fold cross validation for other datasets
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
rnd_state = np.random.RandomState(args.seed)


def split_ids(ids, folds=10):

    if args.dataset == 'COLORS-3':
        assert folds == 1, 'this dataset has train, val and test splits'
        train_ids = [np.arange(500)]
        val_ids = [np.arange(500, 3000)]
        test_ids = [np.arange(3000, 10500)]
    elif args.dataset == 'TRIANGLES':
        assert folds == 1, 'this dataset has train, val and test splits'
        train_ids = [np.arange(30000)]
        val_ids = [np.arange(30000, 35000)]
        test_ids = [np.arange(35000, 45000)]
    else:
        n = len(ids)
        stride = int(np.ceil(n / float(folds)))
        test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all(
            np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        train_ids = []
        for fold in range(folds):
            train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(
                np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

    return train_ids, test_ids


if not args.torch_geom:
    # Unversal data loader and reader (can be used for other graph datasets from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)
    class GraphData(torch.utils.data.Dataset):
        def __init__(self,
                     datareader,
                     fold_id,
                     split):
            self.fold_id = fold_id
            self.split = split
            self.rnd_state = datareader.rnd_state
            self.set_fold(datareader.data, fold_id)

        def set_fold(self, data, fold_id):
            self.total = len(data['targets'])
            self.N_nodes_max = data['N_nodes_max']
            self.num_classes = data['num_classes']
            self.num_features = data['num_features']
            self.idx = data['splits'][fold_id][self.split]
            # use deepcopy to make sure we don't alter objects in folds
            self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])
            self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])
            self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])
            print('%s: %d/%d' % (self.split.upper(), len(self.labels), len(data['targets'])))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            # convert to torch
            return [torch.from_numpy(self.features_onehot[index]).float(),  # node_features
                    torch.from_numpy(self.adj_list[index]).float(),  # adjacency matrix
                    int(self.labels[index])]


    class DataReader():
        '''
        Class to read the txt files containing all data of the dataset.
        Should work for any dataset from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        '''

        def __init__(self,
                     data_dir,  # folder with txt files
                     rnd_state=None,
                     use_cont_node_attr=False,
                     # use or not additional float valued node attributes available in some datasets
                     folds=10):

            self.data_dir = data_dir
            self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
            self.use_cont_node_attr = use_cont_node_attr
            files = os.listdir(self.data_dir)
            data = {}
            nodes, graphs = self.read_graph_nodes_relations(
                list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])

            data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)

            node_labels_file = list(filter(lambda f: f.find('node_labels') >= 0, files))
            if len(node_labels_file) == 1:
                data['features'] = self.read_node_features(node_labels_file[0], nodes, graphs, fn=lambda s: int(s.strip()))
            else:
                data['features'] = None

            data['targets'] = np.array(
                self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0 or f.find('graph_attributes') >= 0, files))[0],
                                    line_parse_fn=lambda s: int(float(s.strip()))))

            if self.use_cont_node_attr:
                data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
                                                       nodes, graphs,
                                                       fn=lambda s: np.array(list(map(float, s.strip().split(',')))))

            features, n_edges, degrees = [], [], []
            for sample_id, adj in enumerate(data['adj_list']):
                N = len(adj)  # number of nodes
                if data['features'] is not None:
                    assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
                if not np.allclose(adj, adj.T):
                    print(sample_id, 'not symmetric')
                n = np.sum(adj)  # total sum of edges
                assert n % 2 == 0, n
                n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
                degrees.extend(list(np.sum(adj, 1)))
                if data['features'] is not None:
                    features.append(np.array(data['features'][sample_id]))

            # Create features over graphs as one-hot vectors for each node
            if data['features'] is not None:
                features_all = np.concatenate(features)
                features_min = features_all.min()
                num_features = int(features_all.max() - features_min + 1)  # number of possible values

            max_degree = np.max(degrees)
            features_onehot = []
            for sample_id, adj in enumerate(data['adj_list']):
                N = adj.shape[0]
                if data['features'] is not None:
                    x = data['features'][sample_id]
                    feature_onehot = np.zeros((len(x), num_features))
                    for node, value in enumerate(x):
                        feature_onehot[node, value - features_min] = 1
                else:
                    feature_onehot = np.empty((N, 0))
                if self.use_cont_node_attr:
                    if args.dataset in ['COLORS-3', 'TRIANGLES']:
                        # first column corresponds to node attention and shouldn't be used as node features
                        feature_attr = np.array(data['attr'][sample_id])[:, 1:]
                    else:
                        feature_attr = np.array(data['attr'][sample_id])
                else:
                    feature_attr = np.empty((N, 0))
                if args.degree:
                    degree_onehot = np.zeros((N, max_degree + 1))
                    degree_onehot[np.arange(N), np.sum(adj, 1).astype(np.int32)] = 1
                else:
                    degree_onehot = np.empty((N, 0))

                node_features = np.concatenate((feature_onehot, feature_attr, degree_onehot), axis=1)
                if node_features.shape[1] == 0:
                    # dummy features for datasets without node labels/attributes
                    # node degree features can be used instead
                    node_features = np.ones((N, 1))
                features_onehot.append(node_features)

            num_features = features_onehot[0].shape[1]

            shapes = [len(adj) for adj in data['adj_list']]
            labels = data['targets']  # graph class labels
            labels -= np.min(labels)  # to start from 0

            classes = np.unique(labels)
            num_classes = len(classes)

            if not np.all(np.diff(classes) == 1):
                print('making labels sequential, otherwise pytorch might crash')
                labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
                for lbl in range(num_classes):
                    labels_new[labels == classes[lbl]] = lbl
                labels = labels_new
                classes = np.unique(labels)
                assert len(np.unique(labels)) == num_classes, np.unique(labels)

            def stats(x):
                return (np.mean(x), np.std(x), np.min(x), np.max(x))

            print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(shapes))
            print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(n_edges))
            print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(degrees))
            print('Node features dim: \t\t%d' % num_features)
            print('N classes: \t\t\t%d' % num_classes)
            print('Classes: \t\t\t%s' % str(classes))
            for lbl in classes:
                print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

            if data['features'] is not None:
                for u in np.unique(features_all):
                    print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))

            N_graphs = len(labels)  # number of samples (graphs) in data
            assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

            # Create train/test sets first
            train_ids, test_ids = split_ids(rnd_state.permutation(N_graphs), folds=folds)

            # Create train sets
            splits = []
            for fold in range(len(train_ids)):
                splits.append({'train': train_ids[fold],
                               'test': test_ids[fold]})

            data['features_onehot'] = features_onehot
            data['targets'] = labels
            data['splits'] = splits
            data['N_nodes_max'] = np.max(shapes)  # max number of nodes
            data['num_features'] = num_features
            data['num_classes'] = num_classes

            self.data = data

        def parse_txt_file(self, fpath, line_parse_fn=None):
            with open(pjoin(self.data_dir, fpath), 'r') as f:
                lines = f.readlines()
            data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
            return data

        def read_graph_adj(self, fpath, nodes, graphs):
            edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
            adj_dict = {}
            for edge in edges:
                node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
                node2 = int(edge[1].strip()) - 1
                graph_id = nodes[node1]
                assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
                if graph_id not in adj_dict:
                    n = len(graphs[graph_id])
                    adj_dict[graph_id] = np.zeros((n, n))
                ind1 = np.where(graphs[graph_id] == node1)[0]
                ind2 = np.where(graphs[graph_id] == node2)[0]
                assert len(ind1) == len(ind2) == 1, (ind1, ind2)
                adj_dict[graph_id][ind1, ind2] = 1

            adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]

            return adj_list

        def read_graph_nodes_relations(self, fpath):
            graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
            nodes, graphs = {}, {}
            for node_id, graph_id in enumerate(graph_ids):
                if graph_id not in graphs:
                    graphs[graph_id] = []
                graphs[graph_id].append(node_id)
                nodes[node_id] = graph_id
            graph_ids = np.unique(list(graphs.keys()))
            for graph_id in graph_ids:
                graphs[graph_id] = np.array(graphs[graph_id])
            return nodes, graphs

        def read_node_features(self, fpath, nodes, graphs, fn):
            node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
            node_features = {}
            for node_id, x in enumerate(node_features_all):
                graph_id = nodes[node_id]
                if graph_id not in node_features:
                    node_features[graph_id] = [None] * len(graphs[graph_id])
                ind = np.where(graphs[graph_id] == node_id)[0]
                assert len(ind) == 1, ind
                assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
                node_features[graph_id][ind[0]] = x
            node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
            return node_features_lst


# NN layers and models
class GraphConv(nn.Module):
    '''
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017) if K<=1
    Chebyshev Graph Convolution Layer according to (M. Defferrard, X. Bresson, and P. Vandergheynst, NIPS 2017) if K>1
    Additional tricks (power of adjacency matrix and weighted self connections) as in the Graph U-Net paper
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 n_relations=1,  # number of relation types (adjacency matrices)
                 K=1,  # GCN is K<=1, else ChebNet
                 activation=None,
                 bnorm=False,
                 adj_sq=False,
                 scale_identity=False):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features * K * n_relations, out_features=out_features)
        self.n_relations = n_relations
        assert K > 0, ('filter scale must be greater than 0', K)
        self.K = K
        self.activation = activation
        self.bnorm = bnorm
        if self.bnorm:
            self.bn = nn.BatchNorm1d(out_features)
        self.adj_sq = adj_sq
        self.scale_identity = scale_identity

    def chebyshev_basis(self, L, X, K):
        if K > 1:
            Xt = [X]
            Xt.append(torch.bmm(L, X))  # B,N,F
            for k in range(2, K):
                Xt.append(2 * torch.bmm(L, Xt[k - 1]) - Xt[k - 2])  # B,N,F
            Xt = torch.cat(Xt, dim=2)  # B,N,K,F
            return Xt
        else:
            # GCN
            assert K == 1, K
            return torch.bmm(L, X)  # B,N,1,F

    def laplacian_batch(self, A):
        batch, N = A.shape[:2]
        if self.adj_sq:
            A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
        A_hat = A
        if self.K < 2 or self.scale_identity:
            I = torch.eye(N).unsqueeze(0).to(args.device)
            if self.scale_identity:
                I = 2 * I  # increase weight of self connections
            if self.K < 2:
                A_hat = A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, data):
        x, A, mask = data[:3]
        # print('in', x.shape, torch.sum(torch.abs(torch.sum(x, 2)) > 0))
        if len(A.shape) == 3:
            A = A.unsqueeze(3)
        x_hat = []

        for rel in range(self.n_relations):
            L = self.laplacian_batch(A[:, :, :, rel])
            x_hat.append(self.chebyshev_basis(L, x, self.K))
        x = self.fc(torch.cat(x_hat, 2))

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)

        x = x * mask  # to make values of dummy nodes zeros again, otherwise the bias is added after applying self.fc which affects node embeddings in the following layers

        if self.bnorm:
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.activation is not None:
            x = self.activation(x)
        return (x, A, mask)


class GCN(nn.Module):
    '''
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 K=1,
                 bnorm=False,
                 n_hidden=0,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False):
        super(GCN, self).__init__()

        # Graph convolution layers
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                K=K,
                                                activation=nn.ReLU(inplace=True),
                                                bnorm=bnorm,
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity) for layer, f in enumerate(filters)]))

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            fc.append(nn.ReLU(inplace=True))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        x = self.fc(x)
        return x


class GraphUnet(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 K=1,
                 bnorm=False,
                 n_hidden=0,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False,
                 shuffle_nodes=False,
                 visualize=False,
                 pooling_ratios=[0.8, 0.8]):
        super(GraphUnet, self).__init__()

        self.shuffle_nodes = shuffle_nodes
        self.visualize = visualize
        self.pooling_ratios = pooling_ratios
        # Graph convolution layers
        self.gconv = nn.ModuleList([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                              out_features=f,
                                              K=K,
                                              activation=nn.ReLU(inplace=True),
                                              bnorm=bnorm,
                                              adj_sq=adj_sq,
                                              scale_identity=scale_identity) for layer, f in enumerate(filters)])
        # Pooling layers
        self.proj = []
        for layer, f in enumerate(filters[:-1]):
            # Initialize projection vectors similar to weight/bias initialization in nn.Linear
            fan_in = filters[layer]
            p = Parameter(torch.Tensor(fan_in, 1))
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(p, -bound, bound)
            self.proj.append(p)
        self.proj = nn.ParameterList(self.proj)

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        # data: [node_features, A, graph_support, N_nodes, label]
        if self.shuffle_nodes:
            # shuffle nodes to make sure that the model does not adapt to nodes order (happens in some cases)
            N = data[0].shape[1]
            idx = torch.randperm(N)
            data = (data[0][:, idx], data[1][:, idx, :][:, :, idx], data[2][:, idx], data[3])

        sample_id_vis, N_nodes_vis = -1, -1
        for layer, gconv in enumerate(self.gconv):
            N_nodes = data[3]

            # TODO: remove dummy or dropped nodes for speeding up forward/backward passes
            # data = (data[0][:, :N_nodes_max], data[1][:, :N_nodes_max, :N_nodes_max], data[2][:, :N_nodes_max], data[3])

            x, A = data[:2]

            B, N, _ = x.shape

            # visualize data
            if self.visualize and layer < len(self.gconv) - 1:
                for b in range(B):
                    if (layer == 0 and N_nodes[b] < 20 and N_nodes[b] > 10) or sample_id_vis > -1:
                        if sample_id_vis > -1 and sample_id_vis != b:
                            continue
                        if N_nodes_vis < 0:
                            N_nodes_vis = N_nodes[b]
                        plt.figure()
                        plt.imshow(A[b][:N_nodes_vis, :N_nodes_vis].data.cpu().numpy())
                        plt.title('layer %d, Input adjacency matrix' % (layer))
                        plt.savefig('input_adjacency_%d.png' % layer)
                        sample_id_vis = b
                        break

            mask = data[2].clone()  # clone as we are going to make inplace changes
            x = gconv(data)[0]  # graph convolution
            if layer < len(self.gconv) - 1:
                B, N, C = x.shape
                y = torch.mm(x.view(B * N, C), self.proj[layer]).view(B, N)  # project features
                y = y / (torch.sum(self.proj[layer] ** 2).view(1, 1) ** 0.5)  # node scores used for ranking below
                idx = torch.sort(y, dim=1)[1]  # get indices of y values in the ascending order
                N_remove = (N_nodes.float() * (1 - self.pooling_ratios[layer])).long()  # number of removed nodes

                # sanity checks
                assert torch.all(
                    N_nodes > N_remove), 'the number of removed nodes must be large than the number of nodes'
                for b in range(B):
                    # check that mask corresponds to the actual (non-dummy) nodes
                    assert torch.sum(mask[b]) == float(N_nodes[b]), (torch.sum(mask[b]), N_nodes[b])

                N_nodes_prev = N_nodes
                N_nodes = N_nodes - N_remove

                for b in range(B):
                    idx_b = idx[b, mask[b, idx[b]] == 1]  # take indices of non-dummy nodes for current data example
                    assert len(idx_b) >= N_nodes[b], (
                        len(idx_b), N_nodes[b])  # number of indices must be at least as the number of nodes
                    mask[b, idx_b[:N_remove[b]]] = 0  # set mask values corresponding to the smallest y-values to 0

                # sanity checks
                for b in range(B):
                    # check that the new mask corresponds to the actual (non-dummy) nodes
                    assert torch.sum(mask[b]) == float(N_nodes[b]), (
                        b, torch.sum(mask[b]), N_nodes[b], N_remove[b], N_nodes_prev[b])
                    # make sure that y-values of selected nodes are larger than of dropped nodes
                    s = torch.sum(y[b] >= torch.min((y * mask.float())[b]))
                    assert s >= float(N_nodes[b]), (s, N_nodes[b], (y * mask.float())[b])

                mask = mask.unsqueeze(2)
                x = x * torch.tanh(y).unsqueeze(2) * mask  # propagate only part of nodes using the mask
                A = mask * A * mask.view(B, 1, N)
                mask = mask.squeeze()
                data = (x, A, mask, N_nodes)

                # visualize data
                if self.visualize and sample_id_vis > -1:
                    b = sample_id_vis
                    plt.figure()
                    plt.imshow(y[b].view(N, 1).expand(N, 2)[:N_nodes_vis].data.cpu().numpy())
                    plt.title('Node ranking')
                    plt.colorbar()
                    plt.savefig('nodes_ranking_%d.png' % layer)
                    plt.figure()
                    plt.imshow(mask[b].view(N, 1).expand(N, 2)[:N_nodes_vis].data.cpu().numpy())
                    plt.title('Pooled nodes (%d/%d)' % (mask[b].sum(), N_nodes_prev[b]))
                    plt.savefig('pooled_nodes_mask_%d.png' % layer)
                    plt.figure()
                    plt.imshow(A[b][:N_nodes_vis, :N_nodes_vis].data.cpu().numpy())
                    plt.title('Pooled adjacency matrix')
                    plt.savefig('pooled_adjacency_%d.png' % layer)
                    print('layer %d: visualizations saved ' % layer)

        if self.visualize and sample_id_vis > -1:
            self.visualize = False  # to prevent visualization for the following batches

        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes
        x = self.fc(x)
        return x


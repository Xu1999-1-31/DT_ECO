import sys
import os
import torch
sys.path.append('../Parsers'); sys.path.append('../'); sys.path.append('../DataTrans') ; sys.path.append('../Model')
import models
import TimingGraphTrans
import PhysicalDataTrans

GNN = models.MultiLayerTimingGNN(3, 32)
G = TimingGraphTrans.LoadTimingGraph('aes_cipher_top')
# G.ndata['padding_mask']=torch.zeros(G.num_nodes(), dtype=torch.float32)
# G.ndata['padding_mask'][0] = 1
# G.ndata['padding_mask'][1] = 1
nf = GNN(G)
print(nf.shape)
# print(G.ndata)
# nodes, nodes_rev = TimingGraphTrans.LoadNodeDict('aes_cipher_top')
# CNN = models.CNN()
# Layout, Padding_Mask, CPath_Padding, Critical_Paths = PhysicalDataTrans.LoadPhysicalData('aes_cipher_top', 512)
# nodes, nodes_rev = TimingGraphTrans.LoadNodeDict('aes_cipher_top')
# Gate_feature = torch.zeros(15, dtype = torch.float32)
# # gf = CNN(Layout)
# MMNN = models.MultiModalNN(3, 64, 32, 32)
# output = MMNN(G, Layout, Padding_Mask[1], Gate_feature)
# print(output)
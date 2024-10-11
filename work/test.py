import sys
import os
sys.path.append('../Parsers'); sys.path.append('../'); sys.path.append('../DataTrans') ; sys.path.append('../Model')
import models
import TimingGraphTrans
import PhysicalDataTrans

GNN = models.MultiLayerTimingGNN(3, 64, 32)
G = TimingGraphTrans.LoadTimingGraph('aes_cipher_top')
# nf = GNN(G)
# print(nf.shape)
# print(G.ndata)
# nodes, nodes_rev = TimingGraphTrans.LoadNodeDict('aes_cipher_top')
# CNN = models.CNN()
Layout, Padding_Mask, CPath_Padding, Critical_Paths = PhysicalDataTrans.LoadPhysicalData('aes_cipher_top', 512)
nodes, nodes_rev = TimingGraphTrans.LoadNodeDict('aes_cipher_top')
Gate_num = [nodes['U2214/A1'], nodes['U2214/Z']]
# gf = CNN(Layout)
MMNN = models.MultiModalNN(3, 64, 32, 32)
output = MMNN(G, Layout, Padding_Mask[1], [2, 3], Gate_num)
print(output)
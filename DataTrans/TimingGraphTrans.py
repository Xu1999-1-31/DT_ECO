import DataBuilder
from sklearn.preprocessing import MinMaxScaler
import dgl
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
import torch
import pickle
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var


# Build Timing Graph
def TimingGraphTrans(design, rebuilt = False, verbose = False):
    print(f'Building {design} Timing Graph.')
    if rebuilt:
        DataBuilder.BuildTimingArc(design)
        DataBuilder.BuildEndPoint(design)
        DataBuilder.BuildPtRpt(design)
    CellArcs, NetArcs = DataBuilder.LoadTimingArc(design)
    EndPoints = DataBuilder.LoadEndPoint(design)
    Critical_Paths = DataBuilder.LoadPtRpt(design)
    # node dict
    nodes = {} # nodes: {'pin1': 0, 'pin2': 1, ...}
    nodes_rev = {} # nodes_rev: {0: 'pin1', 1: 'pin2', ...}
    nodes_feature = {} # nodes_feature: {0: [], 1 : [], ...}
    nodes_slack = {} # nodes_slack: {0: 0, 1: 0, ...}
    nodes_feature_vec = []
    # decide node number and node feature
    for _, cellarc in CellArcs.items():
        if cellarc != None:
            if(cellarc.from_pin not in nodes.keys()):
                nodes[cellarc.from_pin] = len(nodes)
                nodes_rev[len(nodes)-1] = cellarc.from_pin
                nodes_feature[len(nodes)-1] = None
                nodes_slack[len(nodes)-1] = None
            if(cellarc.to_pin not in nodes.keys()):
                nodes[cellarc.to_pin] = len(nodes)
                nodes_rev[len(nodes)-1] = cellarc.to_pin
                nodes_feature[len(nodes)-1] = None
                nodes_slack[len(nodes)-1] = None
    for _, netarc in NetArcs.items():
        if(netarc.from_pin not in nodes.keys()):
            nodes[netarc.from_pin] = len(nodes)
            nodes_rev[len(nodes)-1] = netarc.from_pin
            nodes_feature[len(nodes)-1] = None
            nodes_slack[len(nodes)-1] = None
        if(nodes_feature[nodes[netarc.from_pin]] == None):
            nodes_feature[nodes[netarc.from_pin]] = [netarc.inpin_caps[0], netarc.inpin_caps[1], netarc.isinPinPIPO, 0] # min_cap, max_cap, isPIPO/self loop, isFIFO
        if(netarc.to_pin not in nodes.keys()):
            nodes[netarc.to_pin] = len(nodes)
            nodes_rev[len(nodes)-1] = netarc.to_pin
            nodes_feature[len(nodes)-1] = None
            nodes_slack[len(nodes)-1] = None
        if(nodes_feature[nodes[netarc.to_pin]] == None):
            nodes_feature[nodes[netarc.to_pin]] = [netarc.outpin_caps[0], netarc.outpin_caps[1], netarc.isoutPinPIPO, 1]
    for key, value in nodes_feature.items():
        if value == None:
            nodes_feature[key] = [0, 0, 2, 1] # min_cap, max_cap, isPIPO/self loop, isFIFO 
            value = [0, 0, 2, 1]
        if nodes_rev[key] in EndPoints.keys():
            value.append(EndPoints[nodes_rev[key]])
        else:
            value.append(0)
        nodes_feature_vec.append(value)
    
    scaler = MinMaxScaler()
    nodes_feature_vec = torch.tensor(scaler.fit_transform(nodes_feature_vec), dtype=torch.float32)

    # add edges and edge features
    U_CellArc = []; V_CellArc = []
    U_NetArc = []; V_NetArc = []
    CellArcs_feature_vec = []
    NetArcs_feature_vec = []
    for _, cellarc in CellArcs.items():
        if cellarc != None:
            U_CellArc.append(nodes[cellarc.from_pin]); V_CellArc.append(nodes[cellarc.to_pin])
            CellArcs_feature_vec.append([cellarc.loadCap, cellarc.loadRes, cellarc.effectCap[0], cellarc.effectCap[1], cellarc.outslew[0], cellarc.outslew[1], cellarc.inslew[0], cellarc.inslew[1], cellarc.Delay[0], cellarc.Delay[1]])
    for _, netarc in NetArcs.items():
        U_NetArc.append(nodes[netarc.from_pin]); V_NetArc.append(nodes[netarc.to_pin])
        NetArcs_feature_vec.append([netarc.totalCap, netarc.resistance, netarc.Delay[0], netarc.Delay[1]])
    CellArcs_feature_vec = torch.tensor(scaler.fit_transform(CellArcs_feature_vec), dtype=torch.float32)
    NetArcs_feature_vec = torch.tensor(scaler.fit_transform(NetArcs_feature_vec), dtype=torch.float32)
    # build graph
    data_dict = {
        ('node', 'cellarc', 'node'): (U_CellArc, V_CellArc),
        ('node', 'netarc', 'node'): (U_NetArc, V_NetArc)
    }
    G = dgl.heterograph(data_dict)
    
    G.edata['feature'] = {
        'cellarc': CellArcs_feature_vec, # [loadCap, loadRes, effectCap*2, outslew*2, inslew*2, Delay*2]
        'netarc': NetArcs_feature_vec   # [totalCap, resistance, Delay*2]
    }
    
    G.ndata['feature'] = nodes_feature_vec # [min_cap, max_cap, isPIPO/self loop, isFIFO, EndPoint Slack]

    # Collect Critical Path data
    for path in Critical_Paths:
        path_slack = path.slack
        for pin in reversed(path.Pins):
            if(nodes_slack[nodes[pin.name]] != None):
                nodes_slack[nodes[pin.name]][0] += path_slack
                if(path_slack < nodes_slack[nodes[pin.name]][1]):
                    nodes_slack[nodes[pin.name]][1] = path_slack
            else:
                nodes_slack[nodes[pin.name]] = [path_slack, path_slack, -pin.delay, pin.outtrans] # tns, wns, -pin delay, pin outtrans, rf
                if(pin.rf == 'r'):
                    nodes_slack[nodes[pin.name]].append(0)
                else:
                    nodes_slack[nodes[pin.name]].append(1)
    
    temp = []
    for _, value in nodes_slack.items():
        if(value != None):
            temp.append(value)

    temp = scaler.fit_transform(temp)
    CPath_nf = []
    nodeID = 0
    for key, value in nodes_slack.items():
        if(value != None):
            CPath_nf.append(torch.tensor(temp[nodeID], dtype=torch.float32))
            nodeID += 1
        else:
            CPath_nf.append(torch.tensor([float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]))
    CPath_nf = torch.stack(CPath_nf)
    G.ndata['CPath'] = CPath_nf
    
    # for i, row in enumerate(G.ndata['CPath']):
    #     if ~torch.all(torch.isnan(row)):
    #         print(row)
    
    Save_Dir = Global_var.Trans_Data_Path + 'TimingGraph' 
    if not os.path.exists(Save_Dir):
        os.makedirs(Save_Dir)
    save_path = os.path.join(Save_Dir, design + '_TimingGraph.bin')
    save_graphs(save_path, G)
    save_path = os.path.join(Save_Dir, design + '_nodeDict.sav')
    with open(save_path, 'wb') as f:
        pickle.dump([nodes, nodes_rev], f)
    if verbose:
        print(f'{design} Timing Graph complete!')
    
def LoadTimingGraph(design, rebuild=False, verbose = False):
    # Load Timing Graph
    if verbose:
        print(f'Loading {design} Timing Graph.')
    Save_Dir = Global_var.Trans_Data_Path + 'TimingGraph' 
    save_path = os.path.join(Save_Dir, design + '_TimingGraph.bin')
    if not os.path.exists(save_path) or rebuild:
        TimingGraphTrans(design, True)
    G, _ = load_graphs(save_path)
    if verbose:
        print(f'{design} Timing Graph loaded!')
    return G[0]

def LoadNodeDict(design, rebuild = False, verbose = False):
    # node number and number node
    if verbose:
        print(f'Loading {design} Node Dict.')
    Save_Dir = Global_var.Trans_Data_Path + 'TimingGraph'
    save_path = os.path.join(Save_Dir, design + '_nodeDict.sav')
    if not os.path.exists(save_path) or rebuild:
        TimingGraphTrans(design, True)
    with open(save_path, 'rb') as f:
        nodes, nodes_rev = pickle.load(f)
    if verbose:
        print(f'{design} Node Dict loaded!')
    return nodes, nodes_rev
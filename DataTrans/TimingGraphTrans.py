import DataBuilder
# from sklearn.preprocessing import MinMaxScaler
import dgl
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
import torch
import numpy as np
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
        CellArcs, _ = DataBuilder.BuildTimingArc(design)
        PtCells = DataBuilder.BuildPtCells(design)
        PtNets = DataBuilder.BuildPtNets(design)
        Critical_Paths = DataBuilder.BuildPtRpt(design)
    else:
        CellArcs, _ = DataBuilder.LoadTimingArc(design)
        PtCells = DataBuilder.LoadPtCells(design)
        PtNets = DataBuilder.LoadPtNets(design)
        Critical_Paths = DataBuilder.LoadPtRpt(design)
    timingLib, footprint = DataBuilder.LoadNormalizedTimingLib()
    keyCell_in_Lib = list(timingLib.keys())
    
    nodes = {} # nodes: {'U22': 0, 'U23': 1, ...}
    nodes_rev = {} # nodes_rev: {0: 'U22', 1: 'U23', ...}
    
    U_forward, V_forward, U_backward, V_backward = [] , [], [], [] # start and end point of edges
    
    # Building the Pt Netlist graph, node are the cells
    for _, value in PtNets.items():
        inpins, outpins = [], []
        for inpin in value.inpins:
            if inpin.split('/')[0] not in PtCells.keys(): # filer out Prime Input/Output
                pass
            elif inpin.split('/')[0] not in nodes.keys():
                nodes[inpin.split('/')[0]] = len(nodes)
                nodes_rev[len(nodes)-1] = inpin.split('/')[0]
            if inpin.split('/')[0] in PtCells.keys():
                inpins.append(inpin.split('/')[0])
        for outpin in value.outpins:
            if outpin.split('/')[0] not in PtCells.keys(): # filer out Prime Input/Output
                pass
            elif outpin.split('/')[0] not in nodes.keys():
                nodes[outpin.split('/')[0]] = len(nodes)
                nodes_rev[len(nodes)-1] = outpin.split('/')[0]
            if outpin.split('/')[0] in PtCells.keys():
                outpins.append(outpin.split('/')[0])
        for inpin in inpins:
            for outpin in outpins:
                U_forward.append(nodes[inpin]); V_backward.append(nodes[inpin])
                V_forward.append(nodes[outpin]); U_backward.append(nodes[outpin])
    
    # node feature vector 
    # [max out slew, current gate size, max gate size, TNS, WNS]
    nodes_feature_bidirectional = np.zeros((len(nodes.keys()), 5), dtype=np.float32)
    # [max in slew]
    nodes_feature_forward = np.zeros((len(nodes.keys()), 1), dtype=np.float32)
    # [total cap, total res]
    nodes_feature_backward = np.zeros((len(nodes.keys()), 2), dtype=np.float32)
    # cell type feature, represents the cell type of the node
    nodes_celltype = np.zeros((len(nodes.keys()), 1), dtype=np.int64)
    
    for arc in CellArcs.values():
        if arc != None:
            Cell = arc.from_pin.split('/')[0]
            celltype = keyCell_in_Lib.index(PtCells[Cell].type)
            node_number = nodes[Cell]
            # add cell type, encoded by Int64
            if np.all(nodes_celltype[node_number] == 0):
                nodes_celltype[node_number][0] = celltype
            # add node features, [total cap, total res, max in slew, max out slew, current gate size, max gate size, TNS, WNS]
            if np.all(nodes_feature_bidirectional[node_number] == 0):
                outslew = max(arc.outslew[0], arc.outslew[1])
                inslew = max(arc.inslew[0], arc.inslew[1])
                cellfootprint = timingLib[PtCells[Cell].type].footprint
                gatesize = float(footprint[cellfootprint].index(PtCells[Cell].type))
                maxlength = float(len(footprint[cellfootprint]))
                nodes_feature_bidirectional[node_number] = [outslew, gatesize, maxlength, 0, 0]
                nodes_feature_forward[node_number] = [inslew]
                nodes_feature_backward[node_number] = [arc.loadCap, arc.loadRes]
            else:
                outslew = max(arc.outslew[0], arc.outslew[1], nodes_feature_bidirectional[node_number][0])
                inslew = max(arc.inslew[0], arc.inslew[1], nodes_feature_forward[node_number][0])
                nodes_feature_bidirectional[node_number][0] = outslew
                nodes_feature_forward[node_number][0] = inslew
    
    # Collect Critical Path data
    for path in Critical_Paths:
        path_slack = path.slack
        for arc in path.Cellarcs:
            Cell = arc.name.split('->')[0].split('/')[0]
            node_number = nodes[Cell]
            # Gate wise TNS
            nodes_feature_bidirectional[node_number][3] += path_slack
            # Gate wise WNS
            if path_slack < nodes_feature_bidirectional[node_number][4]:
                nodes_feature_bidirectional[node_number][4] = path_slack
    
    # build graph
    G = dgl.graph((U_forward, V_forward))
    G.ndata['bidirection_feature'] = torch.from_numpy(nodes_feature_bidirectional)
    G.ndata['forward_feature'] = torch.from_numpy(nodes_feature_forward)
    G.ndata['backward_feature'] = torch.from_numpy(nodes_feature_backward)
    G.ndata['celltype'] = torch.from_numpy(nodes_celltype)
    
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
    
def IncrementalUpdate(design, G, nodes):
    CellArcs, PtCells = DataBuilder.BuildCellArc(design)
    Critical_Paths = DataBuilder.LoadPtRpt(design)
    timingLib, footprint = DataBuilder.LoadNormalizedTimingLib()
    keyCell_in_Lib = list(timingLib.keys())
    # node feature vector 
    # [max out slew, current gate size, max gate size, TNS, WNS]
    nodes_feature_bidirectional = np.zeros((len(nodes.keys()), 5), dtype=np.float32)
    # [max in slew]
    nodes_feature_forward = np.zeros((len(nodes.keys()), 1), dtype=np.float32)
    # [total cap, total res]
    nodes_feature_backward = np.zeros((len(nodes.keys()), 2), dtype=np.float32)
    # cell type feature, represents the cell type of the node
    nodes_celltype = np.zeros((len(nodes.keys()), 1), dtype=np.int64)

    for arc in CellArcs.values():
        if arc != None:
            Cell = arc.from_pin.split('/')[0]
            celltype = keyCell_in_Lib.index(PtCells[Cell].type)
            node_number = nodes[Cell]
            # add cell type, encoded by Int64
            if np.all(nodes_celltype[node_number] == 0):
                nodes_celltype[node_number][0] = celltype
            # add node features, [total cap, total res, max in slew, max out slew, current gate size, max gate size, TNS, WNS]
            if np.all(nodes_feature_bidirectional[node_number] == 0):
                outslew = max(arc.outslew[0], arc.outslew[1])
                inslew = max(arc.inslew[0], arc.inslew[1])
                cellfootprint = timingLib[PtCells[Cell].type].footprint
                gatesize = float(footprint[cellfootprint].index(PtCells[Cell].type))
                maxlength = float(len(footprint[cellfootprint]))
                nodes_feature_bidirectional[node_number] = [outslew, gatesize, maxlength, 0, 0]
                nodes_feature_forward[node_number] = [inslew]
                nodes_feature_backward[node_number] = [arc.loadCap, arc.loadRes]
            else:
                outslew = max(arc.outslew[0], arc.outslew[1], nodes_feature_bidirectional[node_number][0])
                inslew = max(arc.inslew[0], arc.inslew[1], nodes_feature_forward[node_number][0])
                nodes_feature_bidirectional[node_number][0] = outslew
                nodes_feature_forward[node_number][0] = inslew
    
    # Collect Critical Path data
    for path in Critical_Paths:
        path_slack = path.slack
        for arc in path.Cellarcs:
            Cell = arc.name.split('->')[0].split('/')[0]
            node_number = nodes[Cell]
            # Gate wise TNS
            nodes_feature_bidirectional[node_number][3] += path_slack
            # Gate wise WNS
            if path_slack < nodes_feature_bidirectional[node_number][4]:
                nodes_feature_bidirectional[node_number][4] = path_slack
    
    G.ndata['bidirection_feature'] = torch.from_numpy(nodes_feature_bidirectional)
    G.ndata['forward_feature'] = torch.from_numpy(nodes_feature_forward)
    G.ndata['backward_feature'] = torch.from_numpy(nodes_feature_backward)
    G.ndata['celltype'] = torch.from_numpy(nodes_celltype)
    
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
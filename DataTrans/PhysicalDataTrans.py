import DataBuilder
import torch
import pickle
import sys
import os
import itertools
import MergeMultiPath
import matplotlib.pyplot as plt
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var

def PortLocationExpander(location):
    x1, y1, x2, y2 = location[0][0], location[0][1], location[1][0], location[1][1]
    location = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    return location

def find_min_bounding_box(bbox1, bbox2):
    all_points = bbox1 + bbox2
    
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)

    return [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]

def PhysicalDataTrans(design, scale, rebuild = False, verbose = False):
    print(f'Building {design} Physical Data.')
    # Building Padding Mask
    if rebuild:
        DataBuilder.BuildCellLayout(design)
        DataBuilder.BuildPtRpt(design)
        DataBuilder.BuildPortData(design)
        DataBuilder.BuildRouteCongestion(design, scale)
        DataBuilder.BuildPinLayout(design)
    CellLayout, CellLocation = DataBuilder.LoadCellLayout(design)
    Critical_Paths = DataBuilder.LoadPtRpt(design)
    Critical_Paths = MergeMultiPath.merge_paths(Critical_Paths)
    PortLocation = DataBuilder.LoadPortData(design)
    for key, value in PortLocation.items():
        PortLocation[key] = PortLocationExpander(value)
    CPath_Padding = []
    # Gate_Padding = []

    if verbose:
        print(f'Building {design} Padding Mask.')
    padding_scale = 8
    Padding_Mask = torch.zeros((int(scale/padding_scale), int(scale/padding_scale)), dtype=torch.float32)
    for path in Critical_Paths:
        CPath_Padding_Mask = torch.zeros((int(scale/padding_scale), int(scale/padding_scale)), dtype=torch.float32)
        for arc in itertools.chain(path.Cellarcs, path.Netarcs):
            # is_cellarc = arc in path.Cellarcs
            pin1, pin2 = arc.name.split('->')
            if '/' not in pin1 and '/' in pin2:
                bbox1 = PortLocation[pin1]
                bbox2 = CellLocation[pin2.split('/')[0]]
            elif '/' not in pin2 and '/' in pin1:
                bbox1 = CellLocation[pin1.split('/')[0]]
                bbox2 = PortLocation[pin2]
            elif '/' not in pin1 and '/' not in pin2:
                bbox1 = PortLocation[pin1]
                bbox2 = PortLocation[pin2]
            else:
                bbox1 = CellLocation[pin1.split('/')[0]]
                bbox2 = CellLocation[pin2.split('/')[0]]

            bbox = find_min_bounding_box(bbox1, bbox2)
            x_min, y_min, x_max, y_max = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]
            block_size = 1.0 / scale * padding_scale
            x_min_idx = max(0, int(x_min // block_size))
            x_max_idx = min(scale, int(torch.ceil(torch.tensor(x_max / block_size)).item()))
            y_min_idx = max(0, int(y_min // block_size))
            y_max_idx = min(scale, int(torch.ceil(torch.tensor(y_max / block_size)).item()))

            Padding_Mask[x_min_idx:x_max_idx, y_min_idx:y_max_idx] = 1
            CPath_Padding_Mask[x_min_idx:x_max_idx, y_min_idx:y_max_idx] = 1

        CPath_Padding_Mask.unsqueeze(0)
        CPath_Padding.append(CPath_Padding_Mask)
        # torch.set_printoptions(threshold=torch.inf)
        # print(Padding_Mask)
    Padding_Mask.unsqueeze(0)
    # for Padding in CPath_Padding:
    #     plt.imshow(Padding.numpy(), cmap='gray')
    #     plt.axis('off')
    #     plt.show() 
    if verbose:
        print(f'{design} Padding Mask complete!')
    
    CellLayout = torch.tensor(CellLayout, dtype=torch.float32)
    Hcongestion, Vcongestion, Layercongestion = DataBuilder.LoadRouteCongestion(design, scale)
    Hcongestion = torch.tensor(Hcongestion, dtype=torch.float32)
    Vcongestion = torch.tensor(Vcongestion, dtype=torch.float32)
    PinLayout = DataBuilder.LoadPinLayout(design)
    PinLayout = torch.tensor(PinLayout, dtype=torch.float32)
    Layout = torch.stack([CellLayout, Hcongestion, Vcongestion, PinLayout], axis=0)
    
    Save_Dir = Global_var.Trans_Data_Path + 'PhysicalData'
    if not os.path.exists(Save_Dir):
        os.makedirs(Save_Dir)
    save_path = os.path.join(Save_Dir, design + '_PhysicalData_' + str(scale) + '.bin')
    torch.save(Layout, save_path)
    save_path = os.path.join(Save_Dir, design + '_PaddingMask_' + str(scale) + '.bin')
    torch.save(Padding_Mask, save_path)
    
    save_path = os.path.join(Save_Dir, design + '_CPathPadding_' + str(scale) + '.sav')
    with open(save_path, 'wb') as f:
        pickle.dump(CPath_Padding, f)
        
    save_path = os.path.join(Save_Dir, design + '_MergedPath.sav')
    with open(save_path, 'wb') as f:
        pickle.dump(Critical_Paths, f)
    
    if verbose:                
        print(f'Building {design} Physical Data complete!')
    
def LoadPhysicalData(design, scale, rebuild = False, verbose = False):
    if verbose:
        print(f'Loading {design} Physical Data.')
    Save_Dir = Global_var.Trans_Data_Path + 'PhysicalData'
    
    if rebuild:
        PhysicalDataTrans(design, scale, True)
    
    save_path = os.path.join(Save_Dir, design + '_PhysicalData_' + str(scale) + '.bin')
    if not os.path.exists(save_path) :
        PhysicalDataTrans(design, scale, True)
    Layout = torch.load(save_path)
    
    save_path = os.path.join(Save_Dir, design + '_PaddingMask_' + str(scale) + '.bin')
    if not os.path.exists(save_path) :
        PhysicalDataTrans(design, scale, True)
    Padding_Mask = torch.load(save_path)
    
    save_path = os.path.join(Save_Dir, design + '_CPathPadding_' + str(scale) + '.sav')
    if not os.path.exists(save_path) :
        PhysicalDataTrans(design, scale, True)
    with open(save_path, 'rb') as f:
        CPath_Padding = pickle.load(f)
    
    save_path = os.path.join(Save_Dir, design + '_MergedPath.sav')
    if not os.path.exists(save_path) :
        PhysicalDataTrans(design, scale, True)
    with open(save_path, 'rb') as f:
        Critical_Paths = pickle.load(f)
    
    if verbose:
        print(f'{design} Physical Data loaded!')
    return Layout, Padding_Mask, CPath_Padding, Critical_Paths

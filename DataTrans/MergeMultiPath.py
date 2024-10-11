import DataBuilder
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var

class MergedPath:
    def __init__(self, path):
        self.Cellarcs = set(path.Cellarcs)  # Using set to avoid duplicates
        self.Cellname_to_Cell = {}
        for arc, cell in zip(path.Cellarcs, path.Cells):
            cellarc_key = arc.name.split('/')[0]  # Name to cell / U222 to NAND
            self.Cellname_to_Cell[cellarc_key] = cell
        self.Netarcs = set(path.Netarcs)
        self.Pins = set(path.Pins)
        self.Cells = set(path.Cells)

    def merge(self, other_path):
        for arc, cell in zip(other_path.Cellarcs, other_path.Cells):
            cellarc_key = arc.name.split('/')[0]
            if cellarc_key not in self.Cellname_to_Cell:
                self.Cellname_to_Cell[cellarc_key] = cell
        self.Cellarcs.update(other_path.Cellarcs)  # Merge without duplicates
        self.Netarcs.update(other_path.Netarcs)
        self.Pins.update(other_path.Pins)
        self.Cells.update(other_path.Cells)

    def __repr__(self):
        cellname_repr = ", ".join([f"{arc}: {self.Cellname_to_Cell[arc]}" for arc in self.Cellname_to_Cell])
        return (f'\nMergedPath(\n  Cellname_to_Cell: {{{cellname_repr}}},\n'
                f'  Cellarcs:{self.Cellarcs},\n  Netarcs: {self.Netarcs},\n  Pins: {self.Pins}\n)')

def find_root(parent, x):
    if parent[x] != x:
        parent[x] = find_root(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    root_x = find_root(parent, x)
    root_y = find_root(parent, y)
    if root_x != root_y:
        if rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        elif rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

def merge_paths(paths):
    n = len(paths)
    parent = list(range(n))
    rank = [0] * n

    # Step 1: Identify overlapping paths based on Cellarcs, if have common cell, merge
    arc_to_path = {}
    for i, path in enumerate(paths):
        for arc in path.Cellarcs:
            if arc.name.split('/')[0] in arc_to_path:
                union(parent, rank, i, arc_to_path[arc.name.split('/')[0]])
            else:
                arc_to_path[arc.name.split('/')[0]] = i

    # Step 2: Group paths by their root
    merged_paths = {}
    for i in range(n):
        root = find_root(parent, i)
        if root not in merged_paths:
            merged_paths[root] = MergedPath(paths[i])
        else:
            merged_paths[root].merge(paths[i])

    return list(merged_paths.values())

# def MergeMultiPath(design):
#     print(f'Merging {design} Multi Paths.')
#     Critical_Paths = DataBuilder.LoadPtRpt(design)
#     print(f'Loaded {len(Critical_Paths)} paths.')
#     merged_paths = merge_paths(Critical_Paths)
#     print(f'Merged {len(merged_paths)} paths.')
    # print(merged_paths)


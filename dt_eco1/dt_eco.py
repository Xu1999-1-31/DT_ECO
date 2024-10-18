from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th
import dgl
from sklearn.preprocessing import MinMaxScaler

def find_module_dir(module_name, search_paths):
    """
    find the target module in the given paths
    :param module_name: target module name
    :param search_paths: list of search paths
    :return: the path which contains the target module or None
    """
    for path in search_paths:
        module_path = os.path.join(path, module_name + ".py")
        if os.path.exists(module_path):
            return path
    return None


import os
import sys
main_script_path = os.path.abspath(sys.argv[0])  # absolute path to the python script being run
main_script_dir = os.path.dirname(main_script_path)  # main script directory
parent_dir = os.path.dirname(main_script_dir) # root directory
search_paths = [
    main_script_dir,
    parent_dir,
    '../'
]
module_dir = find_module_dir('Global_var', search_paths)
if module_dir:
    sys.path.append(module_dir)
else:
    raise ImportError("Project Main dir not found by the 'dt_eco' environment file. Please check the working directory!")
import Global_var
import DataBuilder
import TimingGraphTrans
import PhysicalDataTrans
import Interaction
import ReBuildPtScripts
from DT_ECO_Space import GraphSpace
import time
import math

#DFS Node sort
class GraphDepthSorter:
    def __init__(self, graph):
        self.graph = graph
        self.depth_cache = [None] * graph.num_nodes()  # cache depth of each node

    def calculate_depth_iterative(self, node):
        stack = [(node, 0)]  # (node, current depth)
        max_depth = 0
        while stack:
            current_node, depth = stack.pop()
            if self.depth_cache[current_node] is not None:
                continue
            max_depth = max(max_depth, depth)
            for neighbor in self.graph.successors(current_node):
                if self.depth_cache[neighbor] is None:
                    stack.append((neighbor.item(), depth + 1))

            self.depth_cache[current_node] = max_depth
        return max_depth

    def get_depth_sorted_nodes(self):
        num_nodes = self.graph.num_nodes()
        # calculate depth of each node
        for node in range(num_nodes):
            if self.depth_cache[node] is None:
                self.calculate_depth_iterative(node)
        # sorted nodes by depth
        sorted_nodes = sorted(range(num_nodes), key=lambda n: self.depth_cache[n], reverse=True)
        return sorted_nodes, self.depth_cache

    def sort_sub_nodes_by_depth(self, sub_nodes):
        # Ensure the whole graph is sorted
        self.get_depth_sorted_nodes()
        # sort sub nodes by depth
        sorted_sub_nodes = sorted(sub_nodes, key=lambda n: self.depth_cache[n], reverse=True)
        return sorted_sub_nodes, [self.depth_cache[n] for n in sorted_sub_nodes]

def report_time(elapsed_time, stage):
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f'Runtime for {stage}: {hours} hours, {minutes} minutes, {seconds:.2f} seconds')

class DT_ECO(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode: Optional[str] = None, current_design: Optional[str] = 'aes_cipher_top'):
        start_time = time.time()
        super(DT_ECO, self).__init__()
        print(f'Initializing DT_ECO environment for {current_design}')
        self.render_mode = render_mode  # online monitor
        self.current_design = current_design  # the design to be optimized
        self.current_step = 0
        
        # Rebuild Pt Timing Arc scripts
        ReBuildPtScripts.ReBuildPtScripts(self.current_design)
        
        # Load the timing graph
        self.graph = TimingGraphTrans.LoadTimingGraph(self.current_design, True)
        
        # Load the node dictionary (cell -> number, number -> cell)
        self.nodes, self.nodes_rev = TimingGraphTrans.LoadNodeDict(self.current_design)
        
        # list to record all cell sized, for ECO scripts writing
        self.sizedCellList = []
        
        # DFS node sorter
        self.sorter = GraphDepthSorter(self.graph)
        
        # target node of one step
        self.target_cells = []
        
        # subgraph extraction
        CriticalPaths = DataBuilder.LoadPtRpt(self.current_design)
        Cells_num = set()
        for path in CriticalPaths:
            for cellarc in path.Cellarcs:
                cellname = cellarc.name.split('->')[0].split('/')[0]
                Cells_num.add(self.nodes[cellname])
        self.Cells_num = list(Cells_num) # target node number, cells on critical paths
        self.sorted_cells_num, _ = self.sorter.sort_sub_nodes_by_depth(self.Cells_num)
        # sub graph of current step target gates
        # self.subgraph = dgl.khop_in_subgraph(self.graph, self.sorted_cells_num, k=3)[0]
        self.cells_one_iteration = 20
        self.total_iteration = math.ceil(len(Cells_num)/self.cells_one_iteration)
        # delta driving strength list before
        self.delta_strength_list = np.zeros((self.total_iteration, self.cells_one_iteration), dtype=np.float32) # sequence length, embedding dim
        
        # Load the timing library and cell footprint
        self.timingLib, self.footprint = DataBuilder.LoadNormalizedTimingLib()
        # max_length = max(len(v) for v in self.footprint.values())
        
        '''
        the observation space contains six parameters:
        gate_sizes represent the sizes currently chosen, 
        timing graph contains the current step timing graph,
        gate_features are preprocessed timing features for the target gate of the step,
        layout is the physical layout of the design,
        padding_mask is the padding mask of the current critical path.
        '''
        
        self.observation_space = spaces.Dict({
            'delta_strength': spaces.Box(
                low=0, high=1,  # gate sizes strength change selected before, -14 to 14
                shape=(self.total_iteration, self.cells_one_iteration), dtype=np.float32
            ),
            'timing_graph': GraphSpace(
                self.graph
            ),
        })
        # driving strength change, from -1 to 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.cells_one_iteration, 1), dtype=np.float32) # delta driven strength
        self.reward_space = spaces.Box(
            low=np.array([-100000, 0]),
            high=np.array([0, 100000]),
            dtype=np.float32,
        )
        # reward space [[low[0]:high[0]], [low[1]:high[1]]]
        self.reward_dim = 2
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        report_time(elapsed_time, 'Initialization')
        
        print(f'Initialized DT_ECO environment with observation space: \n<{self.observation_space}>\nand action space: \n<{self.action_space}>')
        
    def render(self): # needed
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.sizedCellList = []
        CriticalPaths = DataBuilder.LoadPtRpt(self.current_design)
        Cells_num = set()
        for path in CriticalPaths:
            for cellarc in path.Cellarcs:
                cellname = cellarc.name.split('->')[0].split('/')[0]
                Cells_num.add(self.nodes[cellname])
        self.Cells_num = list(Cells_num)
        self.sorted_cells_num, _ = self.sorter.sort_sub_nodes_by_depth(self.Cells_num)
        # self.subgraph = dgl.khop_in_subgraph(self.graph, self.sorted_cells_num, k=3)[0]
        self.timingLib, self.footprint = DataBuilder.LoadNormalizedTimingLib()
        self.delta_strength_list = np.zeros((self.total_iteration, self.cells_one_iteration), dtype=np.float32)
        self.target_cells = []
        return self._get_obs(), {}
    
    def _get_obs(self):
        #Combine timing graph, delta strength into a single observation.
        scaler = MinMaxScaler()
        delta_strength = th.from_numpy(self.delta_strength_list)  # chosen strength
        timing_graph = self.graph.clone()  # get sub graph
        timing_graph.ndata['bidirection_feature'] = th.from_numpy(scaler.fit_transform(timing_graph.ndata['bidirection_feature'].numpy()))
        timing_graph.ndata['forward_feature'] = th.from_numpy(scaler.fit_transform(timing_graph.ndata['forward_feature'].numpy()))
        timing_graph.ndata['backward_feature'] = th.from_numpy(scaler.fit_transform(timing_graph.ndata['backward_feature'].numpy()))
        timing_graph = self._add_padding_mask(timing_graph)
        # timing_graph = dgl.khop_in_subgraph(self.graph, self.sorted_cells_num + self.target_cells, k=3)[0]
        timing_graph = dgl.khop_in_subgraph(timing_graph, self.Cells_num, k=3)[0]
        obs = {
            'delta_strength': delta_strength,
            'timing_graph': timing_graph,
        }
        return obs
    
    def _pop_target_cells(self):
        """
        pop the cells for one iteration  sorted nodes
        """
        top_k = self.sorted_cells_num[:self.cells_one_iteration]
        result = top_k
        self.sorted_cells_num = self.sorted_cells_num[self.cells_one_iteration:]

        return result
    
    def _get_tns_drc(self, inline=False, ECO=False):
        if ECO and inline:
            raise ValueError("ECO and inline cannot both be True at the same time.")
        if inline:
            wns, tns = DataBuilder.BuildGlobalTimingData(self.current_design+'_inline')
            drc = DataBuilder.BuildDrcNumber(self.current_design)
        elif ECO:
            wns, tns = DataBuilder.BuildGlobalTimingData(self.current_design+'_eco')
            drc = DataBuilder.BuildDrcNumber(self.current_design+'_eco')
        else:
            wns, tns = DataBuilder.BuildGlobalTimingData(self.current_design)
            drc = DataBuilder.BuildDrcNumber(self.current_design)
        return [tns, drc]
    
    def _get_gate_size(self, action): #
        # gate_sizes = np.zeros(len(self.CriticalPaths[self.current_path_index].Cellname_to_Cell.keys()), dtype=np.float32)  # total number of gates on paths
        origin_gate_size = self.graph.ndata['bidirection_feature'][self.target_cells, 1].numpy() # current gate sizes 8 for example
        max_gate_size = self.graph.ndata['bidirection_feature'][self.target_cells, 2].numpy() - 1
        min_gate_size = np.zeros(max_gate_size.shape)
        action = action[0, :len(origin_gate_size)]
        gate_size = np.copy(origin_gate_size)
        
        mask_neg = action < 0
        gate_size[mask_neg] = origin_gate_size[mask_neg] + (origin_gate_size[mask_neg] - min_gate_size[mask_neg]) * action[mask_neg]
        mask_pos = action >= 0
        gate_size[mask_pos] = origin_gate_size[mask_pos] + (max_gate_size[mask_pos] - origin_gate_size[mask_pos]) * action[mask_pos]

        
        gate_size = np.round(gate_size).astype(int)
        return gate_size
    
    def _get_sized_cellDict(self, action):
        sizedCell = {}
        chosen_sizes = self._get_gate_size(action)
        gate_types = self.graph.ndata['celltype'][self.target_cells]
        # gate_type: int64, represents the gate number in the lib; target cell: float32, represents the node number
        # chosen size: int, represents the chosed number in the footprint dict
        for gate_type, target_cell, chosen_size in zip(gate_types, self.target_cells, chosen_sizes):
            origin_type = list(self.timingLib.keys())[gate_type] # NANDV2
            footprint = self.timingLib[origin_type].footprint
            chosen_type = self.footprint[footprint][chosen_size] # NANDV4
            cellname = self.nodes_rev[target_cell]
            if origin_type != chosen_type:
                sizedCell[cellname] = [origin_type, chosen_type]
        return sizedCell
    
    def _add_padding_mask(self, g):
        padding_mask = th.zeros(g.num_nodes(), dtype=th.bool)
        padding_mask[self.target_cells] = True
        g.ndata['padding_mask'] = padding_mask
        return g
        
    def step(self, action):
        action = np.transpose(action, (1, 0)) # (10, 1) to (1, 10)
        if self.current_step < self.total_iteration:
            self.target_cells = self._pop_target_cells()
            self.delta_strength_list[self.current_step, :action.shape[1]] = action[0, :] # expand current action
            changed_cell_dict = self._get_sized_cellDict(action)
            self.sizedCellList.append(changed_cell_dict)
            if self.current_step != self.total_iteration -1:
                # change netlist
                if self.current_step == 0: # first path
                    Interaction.VerilogInlineChange(self.current_design, changed_cell_dict, Incremental=False)
                else:
                    Interaction.VerilogInlineChange(self.current_design, changed_cell_dict, Incremental=True) 
                # run pt
                Interaction.VerilogInline_PT_Iteration(self.current_design)
                TimingGraphTrans.IncrementalUpdate(self.current_design + '_inline', self.graph, self.nodes)
                vec_reward = self._get_tns_drc(inline=True, ECO=False)
                
                # info = {'message':'One step of cells sized sucessfully'}
            else:
                # change netlist and write ECO gate sizing scripts
                Interaction.VerilogInlineChange(self.current_design, changed_cell_dict, Incremental=True)
                Interaction.Write_Incremental_ECO_Scripts(self.current_design, self.sizedCellList)
                
                # run icc2 and pt
                # Interaction.ECO_PRPT_Iteration(self.current_design)
                # TimingGraphTrans.IncrementalUpdate(self.current_design + '_eco', self.graph, self.nodes)
                # vec_reward = self._get_tns_drc(inline=False, ECO=True)
                
                # run pt only
                Interaction.VerilogInline_PT_Iteration(self.current_design)
                TimingGraphTrans.IncrementalUpdate(self.current_design + '_inline', self.graph, self.nodes)
                vec_reward = self._get_tns_drc(inline=True, ECO=False)
                # info = {'message':'One PR Episode completed sucessfully'}
            done = False
            self.current_step += 1
            # self.subgraph = dgl.khop_in_subgraph(self.graph, self.sorted_cells_num, k=3)[0]
        else:
            self.reset()
            vec_reward = [0, 0]
            done = True
            self.current_step = 0
        vec_reward = [0 if vec_reward[0] == 0 else -1/vec_reward[0], vec_reward[1]]
        # truncated = False
    
        return self._get_obs(), vec_reward, done

if __name__ == "__main__": 
    start_time = time.time()
    env = DT_ECO()
    for i in range(10):
        env.render()
        action = env.action_space.sample() # random action space sampling
        current_state, vec_reward, done = env.step(action)
        # np.set_printoptions(edgeitems=10, threshold=20, precision=4, suppress=False, linewidth=np.inf)
        print(f'step {i} reward: {vec_reward}')
        if done:
            env.reset()
    env.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    report_time(elapsed_time, 'One Eposide')
    # time = elapsed_time/10
    # print(f'average time per step:{time}')
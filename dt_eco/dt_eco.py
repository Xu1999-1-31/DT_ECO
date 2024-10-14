from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th
import os
import sys
main_script_path = os.path.abspath(sys.argv[0])  # absolute path to the python script being run
main_script_dir = os.path.dirname(main_script_path)  # main script directory
parent_dir = os.path.dirname(main_script_dir) # root directory
sys.path.append(parent_dir)
sys.path.append('../')
import Global_var
import DataBuilder
import TimingGraphTrans
import PhysicalDataTrans
import Interaction
import ReBuildPtScripts
from DT_ECO_Space import GraphSpace
import time


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
        
        self.current_gate_index = 0 # the number of the gate on the path
        self.current_path_index = 0 # the number of the path
        
        # Rebuild Pt Timing Arc scripts
        ReBuildPtScripts.ReBuildPtScripts(self.current_design)
        
        # Load the timing graph
        self.graph = TimingGraphTrans.LoadTimingGraph(self.current_design, True)
        
        # Load the node dictionary (pin -> number, number -> pin)
        self.nodes, self.nodes_rev = TimingGraphTrans.LoadNodeDict(self.current_design)
        
        # Load the physical data (layout, padding mask, critical path)
        self.Layout, self.Padding_Mask, self.Cpath_Padding, self.CriticalPaths = PhysicalDataTrans.LoadPhysicalData(self.current_design, 512, False)

        # Total number of cells
        self.num_cells = 0
        # steps needed for one eposide
        self.steps_needed = 0
        for i in range(len(self.CriticalPaths)):
            self.num_cells += len(self.CriticalPaths[i].Cellname_to_Cell.keys())
            self.steps_needed += len(self.CriticalPaths[i].Cellname_to_Cell.keys()) + 1
        
        # Load the timing library and cell footprint
        self.timingLib, self.footprint = DataBuilder.LoadNormalizedTimingLib()
        
        # Load Pt Cells to distinguish the in and out pin of the cell
        self.PtCells = DataBuilder.LoadPtCells(self.current_design)
        
        # Set the action space based on the maximum footprint length
        max_length = max(len(v) for v in self.footprint.values()) # max gate type
        self.gate_size_list = [i/max_length for i in range(max_length)]  # action space
        
        # Dynamic creation of observation space based on the loaded data
        num_nodes = self.graph.number_of_nodes()  # dynamically get the number of nodes
        node_feature_dim = self.graph.ndata['feature'].shape[1]  # dynamically get node feature dimension
        layout_shape = self.Layout.shape  # dynamically get layout shape (should be 4 dimensions)
        
        scale = layout_shape[1]  # assume layout has shape (channels, height, width)
        
        self.max_cells = max(len(path.Cellname_to_Cell.keys()) for path in self.CriticalPaths)
        self.gate_sizes = np.zeros((len(self.CriticalPaths), self.max_cells), dtype=np.float32)
        self.sizedCellList = [] # all dicts with sized cells
        self.Gate_feature = th.zeros(15, dtype=th.float32)
        self._add_graph_padding()

        
        '''
        the observation space contains six parameters:
        gate_sizes represent the sizes currently chosen, 
        timing graph contains the current step timing graph,
        gate_features are preprocessed timing features for the target gate of the step,
        layout is the physical layout of the design,
        padding_mask is the padding mask of the current critical path.
        '''
        
        self.observation_space = spaces.Dict({
            'gate_sizes': spaces.Box(
                low=0, high=14,  # gate sizes are discrete values between 0 and 14
                shape=(len(self.CriticalPaths), self.max_cells), dtype=np.float32
            ),
            'timing_graph': GraphSpace(
                self.graph
            ),
            'gate_features': spaces.Box(
                low=0, high=10,  # sum of gate node 'CPath' feature and mean of gate edge feature
                shape=(15,), dtype=np.float32
            ),
            'layout': spaces.Box(
                low=0, high=1,  # physical features are normalized [0, 1]
                shape=layout_shape, dtype=np.float32
            ),
            'padding_mask': spaces.Box(
                low=0, high=1,  # padding masks are binary [0, 1]
                shape=(1, scale // 8, scale // 8),
                dtype=np.float32
            ),
        })
        
        self.action_space = spaces.Discrete(max_length)
        # self.reward_space = spaces.Box(
        #     low=np.array([-100000, 0]),
        #     high=np.array([0, 100000]),
        #     dtype=np.float32,
        # )
        # # reward space [[low[0]:high[0]], [low[1]:high[1]]]
        # self.reward_dim = 2
        
        self.reward_space = spaces.Box(
            low=np.array([0]),
            high=np.array([1]),
            dtype=np.float32,
        )
        # reward space [[low[0]:high[0]], [low[1]:high[1]]]
        self.reward_dim = 1
        
        self.inline=False # if inline Verilog timing or not
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        report_time(elapsed_time, 'Initialization')
        
        print(f'Initialized DT_ECO environment with observation space: \n<{self.observation_space}>\nand action space: \n<{self.action_space}>')
        
    def _process_gate_features(self):
        current_gate = list(self.CriticalPaths[self.current_path_index].Cellname_to_Cell.keys())[self.current_gate_index]
        nf_index = []
        ef_index = []
        for outpin in self.PtCells[current_gate].outpins:
            nf_index.append(self.nodes[current_gate + '/' + outpin])
            for inpin in self.PtCells[current_gate].inpins:
                nf_index.append(self.nodes[current_gate + '/' + inpin])
                # avoid edge not exist, probably arc from D to Q
                if self.graph.has_edges_between(self.nodes[current_gate + '/' + inpin], self.nodes[current_gate + '/' + outpin], etype='cellarc'):
                    ef_index.append(self.graph.edge_ids(self.nodes[current_gate + '/' + inpin], self.nodes[current_gate + '/' + outpin], etype='cellarc'))
                else:
                    # print(current_gate + '/' + inpin, current_gate + '/' + outpin)
                    pass
        Gate_nf = th.sum(th.nan_to_num(self.graph.ndata['CPath'][nf_index], nan=0.0), dim=0)
        Gate_ef = th.mean(self.graph.edata['feature'][('node', 'cellarc', 'node')][ef_index], dim=0)
        self.Gate_feature = th.cat([Gate_nf, Gate_ef], dim=0)
    
    # add the padding mask for the nodes on critical path
    def _add_graph_padding(self):
        node_numbers = []
        pins = self.CriticalPaths[self.current_path_index].Pins
        for pin in pins:
            nodes = self.nodes[pin.name]
            node_numbers.append(nodes)
        padding_mask = th.zeros(self.graph.num_nodes(), dtype=th.float32)
        padding_mask[node_numbers] = 1 # 1 means nodes on critical path
        self.graph.ndata['padding_mask'] = padding_mask
        
    def render(self): # needed
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_gate_index = 0 # the number of the gate on the path
        self.current_path_index = 0 # the number of the path
        self.chosen_sizes = [] # the sizes of the gates on the path
        self.sizedCellList = [] # all dicts with sized cells
        self.graph = TimingGraphTrans.LoadTimingGraph(self.current_design, False)
        self.nodes, self.nodes_rev = TimingGraphTrans.LoadNodeDict(self.current_design)
        self.Layout, _, self.Cpath_Padding, self.CriticalPaths = PhysicalDataTrans.LoadPhysicalData(self.current_design, 512, False)
        self.max_cells = max(len(path.Cellname_to_Cell.keys()) for path in self.CriticalPaths)
        self.gate_sizes = np.zeros((len(self.CriticalPaths), self.max_cells), dtype=np.float32)
        self.Gate_feature = th.zeros(15, dtype=th.float32)
        self._add_graph_padding()
        return self._get_obs(), {}
    
    def _get_obs(self, ECO=False):
        #Combine gate sizes, timing graph, layout, and padding mask into a single observation.
        gate_sizes = th.from_numpy(self.gate_sizes)  # gate sizes
        timing_graph_features = self.graph  # get graph
        gate_features = self.Gate_feature  # get gate features
        
        layout = self.Layout  # physical layout
        if ECO:
            padding_mask = self.Padding_Mask
        else:
            padding_mask = self.Cpath_Padding[self.current_path_index]
        # node_numbers = self.node_numbers
        
        obs = {
            'gate_sizes': gate_sizes,
            'timing_graph': timing_graph_features,
            'gate_features': gate_features,
            'layout': layout,
            'padding_mask': padding_mask,
        }
        return obs
    
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
    
    def step(self, action):  
        # new episode begin
        if self.current_path_index >= len(self.CriticalPaths):
            self.current_path_index = 0
            self.sizedCellList
            self.inline = False
             
        if self.current_gate_index < len(self.CriticalPaths[self.current_path_index].Cellname_to_Cell.keys()): # one action not finished // Cellname_to_Cell: U222 -> NAND
            self.chosen_sizes.append(self.gate_size_list[action])  # append the gate size list
            self.gate_sizes = np.zeros((len(self.CriticalPaths), self.max_cells), dtype=np.float32)
            
            # if the action finished and episode not: one critical merged path is sized
            if len(self.chosen_sizes) == len(self.CriticalPaths[self.current_path_index].Cellname_to_Cell.keys()) and self.current_path_index != len(self.CriticalPaths) - 1:
                start_time = time.time()
                # inline verilog chage
                changed_cell_dict = self._get_sized_cellDict()
                self.sizedCellList.append(changed_cell_dict)
                if self.current_path_index == 0: # first path
                    Interaction.VerilogInlineChange(self.current_design, changed_cell_dict, Incremental=False)
                    self.inline = True
                else:
                    Interaction.VerilogInlineChange(self.current_design, changed_cell_dict, Incremental=True) 
                    
                # # run pt
                Interaction.VerilogInline_PT_Iteration(self.current_design)
                # Timing Graph update (inline)
                self.graph = TimingGraphTrans.LoadTimingGraph(self.current_design+'_inline', True)               
                self.nodes, self.nodes_rev = TimingGraphTrans.LoadNodeDict(self.current_design+'_inline')
                
                done = False
                info = {'message':'One critical merged path sized sucessfully'}
                end_time = time.time()
                report_time(end_time - start_time, 'One Path Inline Sizing')
                vec_reward = self._get_tns_drc(self.inline, False)
                
            # if the action finished and episode finished: the whole design is sized
            elif len(self.chosen_sizes) == len(self.CriticalPaths[self.current_path_index].Cellname_to_Cell.keys()) and self.current_path_index == len(self.CriticalPaths) - 1:
                start_time  = time.time()
                # write changelists
                changed_cell_dict = self._get_sized_cellDict()
                self.sizedCellList.append(changed_cell_dict)
                Interaction.VerilogInlineChange(self.current_design, changed_cell_dict, Incremental=True)
                Interaction.Write_Incremental_ECO_Scripts(self.current_design, self.sizedCellList)
                
                # run icc2 and pt
                # Interaction.ECO_PRPT_Iteration(self.current_design)
                # Timing Graph update (ECO)
                self.graph = TimingGraphTrans.LoadTimingGraph(self.current_design+'_eco', True)
                # Physical Data update (ECO)
                self.Layout, self.Padding_Mask, self.Cpath_Padding, self.CriticalPaths = PhysicalDataTrans.LoadPhysicalData(self.current_design+'_eco', 512, True)
                
                done = True
                info = {'message':'One PR Episode completed sucessfully'}
                end_time = time.time()
                report_time(end_time - start_time, 'One PR Incremental Sizing')
                vec_reward = self._get_tns_drc(False, True)
                
            else:
                done = False
                vec_reward = self._get_tns_drc(self.inline)
                info = {}
            self._process_gate_features()
            # get the gate feature for the current gate
            self.current_gate_index += 1
        else: # new action begin
            self.current_gate_index = 0
            self.chosen_sizes = []
            self.current_path_index += 1
            self.sizedCellList = []
            self.Gate_feature = th.zeros(15, dtype=th.float32)
            done = False
            vec_reward = self._get_tns_drc(self.inline)
            info = {}
        
        vec_reward = -1/vec_reward[0]  # only return -tns
        # get current graph padding
        self._add_graph_padding()
    
        return self._get_obs(done), vec_reward, done, False, info

    def _get_gate_sizes(self):
        gate_sizes = np.zeros(len(self.CriticalPaths[self.current_path_index].Cellname_to_Cell.keys()), dtype=np.int32)  # total number of gates on paths
        gate_sizes[:len(self.chosen_sizes)] = [val + self.gate_size_list[1] for val in self.chosen_sizes] # plus 1/max_length to avoid 0, since 0 means gate remain its original size
        self.gate_sizes[self.current_path_index, :len(gate_sizes)] = gate_sizes # update the gate sizes observation
        return gate_sizes
    
    def _get_sized_cellDict(self):
        sizedCell = {}
        for (cellname, celltype), chosedsize in zip(self.CriticalPaths[self.current_path_index].Cellname_to_Cell.items(), self._get_gate_sizes()):
            originalsize = celltype # NANDX2
            cellfootprint = self.timingLib[celltype].footprint # Cell Timing Lib Structure
            maxsize = len(self.footprint[cellfootprint])* self.gate_size_list[1] # chosed size should be less than maxsize
            if chosedsize == 0:
                pass # 0 means remain the original size
            elif chosedsize <= maxsize:
                chosedsize = self.gate_size_list.index(chosedsize - self.gate_size_list[1]) # resore the size to the index
                newsize = self.footprint[cellfootprint][chosedsize]
                sizedCell[cellname] = [originalsize, newsize]
            else:
                pass # invalid size
        return sizedCell

if __name__ == "__main__": 
    start_time = time.time()
    env = DT_ECO()
    env.reset()
    # for _ in range(env.steps_needed):
    #     env.render()
    #     action = env.action_space.sample() # random action space sampling
    #     # print(f'action: {action}#{type(action)}')
    #     current_state, vec_reward, done, truncated, info = env.step(action)
    #     # np.set_printoptions(edgeitems=10, threshold=20, precision=4, suppress=False, linewidth=np.inf)
    #     # print(current_state['gate_sizes'])
    #     print(f'reward: {vec_reward}')
    #     if done:
    #         env.reset()
    env.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    report_time(elapsed_time, 'One Eposide')
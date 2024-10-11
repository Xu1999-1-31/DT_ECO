from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
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
        
        # Rebuild Pt Timing Arc scripts
        ReBuildPtScripts.ReBuildPtScripts(self.current_design)
        
        # Load the timing graph
        self.graph = TimingGraphTrans.LoadTimingGraph(self.current_design, False)
        
        # Load the node dictionary (pin -> number, number -> pin)
        self.nodes, self.nodes_rev = TimingGraphTrans.LoadNodeDict(self.current_design)
        
        # Load the physical data (layout, padding mask, critical path)
        self.Layout, _, self.Cpath_Padding, self.CriticalPaths = PhysicalDataTrans.LoadPhysicalData(self.current_design, 512, False)

        # Total number of cells
        self.num_cells = 0
        # steps needed for one eposide
        self.steps_needed = 0
        for i in range(len(self.CriticalPaths)):
            self.num_cells += len(self.CriticalPaths[i].Cellname_to_Cell.keys())
            self.steps_needed += len(self.CriticalPaths[i].Cellname_to_Cell.keys()) + 1
        
        # Load the timing library and cell footprint
        self.timingLib, self.footprint = DataBuilder.LoadNormalizedTimingLib()
        
        # Set the action space based on the maximum footprint length
        max_length = max(len(v) for v in self.footprint.values()) # max gate type
        self.action_space_list = [i for i in range(max_length)]  # action space
        
        # Dynamic creation of observation space based on the loaded data
        num_nodes = self.graph.number_of_nodes()  # dynamically get the number of nodes
        node_feature_dim = self.graph.ndata['feature'].shape[1]  # dynamically get node feature dimension
        layout_shape = self.Layout.shape  # dynamically get layout shape (should be 4 dimensions)
        
        scale = layout_shape[1]  # assume layout has shape (channels, height, width)
        
        self.max_cells = max(len(path.Cellname_to_Cell.keys()) for path in self.CriticalPaths)
        self.gate_sizes = np.zeros((len(self.CriticalPaths), self.max_cells), dtype=np.int32)
        self.sizedCellList = [] # all dicts with sized cells
        
        self.observation_space = spaces.Dict({
            'gate_sizes': spaces.Box(
                low=0, high=14,  # gate sizes are discrete values between 0 and 14
                shape=(len(self.CriticalPaths), self.max_cells), dtype=np.int32
            ),
            'timing_graph': spaces.Box(
                low=0, high=1,  # node features are normalized [0, 1]
                shape=(num_nodes, node_feature_dim), dtype=np.float32
            ),
            'layout': spaces.Box(
                low=0, high=1,  # physical features are normalized [0, 1]
                shape=layout_shape, dtype=np.float32
            ),
            'padding_mask': spaces.Box(
                low=0, high=1,  # padding masks are binary [0, 1]
                shape=(len(self.Cpath_Padding), scale // 8, scale // 8),
                dtype=np.float32
            ),
        })
        
        self.action_space = spaces.Discrete(max_length)
        self.reward_space = spaces.Box(
            low=np.array([-100000, 0]),
            high=np.array([0, 100000]),
            dtype=np.float32,
        )
        # reward space [[low[0]:high[0]], [low[1]:high[1]]]
        self.reward_dim = 2
        
        self.inline=False # if inline Verilog timing or not
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        report_time(elapsed_time, 'Initialization')
        
        print(f'Initialized DT_ECO environment with observation space: \n<{self.observation_space}>\nand action space: \n<{self.action_space}>')
        
    def render(self): # needed
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_gate_index = 0 # the number of the gate on the path
        self.current_path_index = 0 # the number of the path
        self.chosen_sizes = []
        self.sizedCellList = []
        self.graph = TimingGraphTrans.LoadTimingGraph(self.current_design, False)
        self.nodes, self.nodes_rev = TimingGraphTrans.LoadNodeDict(self.current_design)
        self.Layout, _, self.Cpath_Padding, self.CriticalPaths = PhysicalDataTrans.LoadPhysicalData(self.current_design, 512, False)
        self.max_cells = max(len(path.Cellname_to_Cell.keys()) for path in self.CriticalPaths)
        self.gate_sizes = np.zeros((len(self.CriticalPaths), self.max_cells), dtype=np.int32)
        return self._get_obs(), {}
    
    def _get_obs(self):
        #Combine gate sizes, timing graph, layout, and padding mask into a single observation.
        gate_sizes = self.gate_sizes  # gate sizes
        timing_graph_features = self.graph.ndata['feature'].numpy()  # get node features
        layout = self.Layout.numpy()  # physical layout
        padding_mask = [mask.numpy() for mask in self.Cpath_Padding]  # padding mask
        
        obs = {
            "gate_sizes": gate_sizes,
            "timing_graph": timing_graph_features,
            "layout": layout,
            "padding_mask": padding_mask,
        }
        return obs
    
    def _get_tns_drc(self, inline=False, ECO=False):
        if ECO and inline:
            raise ValueError("ECO and inline cannot both be True at the same time.")
        if inline:
            _, tns = DataBuilder.BuildGlobalTimingData(self.current_design+'_inline')
            drc = DataBuilder.BuildDrcNumber(self.current_design)
        elif ECO:
            _, tns = DataBuilder.BuildGlobalTimingData(self.current_design+'_eco')
            drc = DataBuilder.BuildDrcNumber(self.current_design+'_eco')
        else:
            _, tns = DataBuilder.BuildGlobalTimingData(self.current_design)
            drc = DataBuilder.BuildDrcNumber(self.current_design)
        return [tns, drc]
    
    def step(self, action):  
        if self.current_gate_index < len(self.CriticalPaths[self.current_path_index].Cellname_to_Cell.keys()): # one action not finished // Cellname_to_Cell: U222 -> NAND
            self.chosen_sizes.append(self.action_space_list[action])  # append the gate size list
            self._get_gate_sizes()
            
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
                Interaction.ECO_PRPT_Iteration(self.current_design)
                # Timing Graph update (ECO)
                self.graph = TimingGraphTrans.LoadTimingGraph(self.current_design+'_eco', True)
                # Physical Data update (ECO)
                self.Layout, _, self.Cpath_Padding, self.CriticalPaths = PhysicalDataTrans.LoadPhysicalData(self.current_design+'_eco', 512, True)
                
                done = True
                info = {'message':'One PR Episode completed sucessfully'}
                end_time = time.time()
                report_time(end_time - start_time, 'One PR Incremental Sizing')
                vec_reward = self._get_tns_drc(False, True)
                
            else:
                done = False
                vec_reward = self._get_tns_drc(self.inline)
                info = {}
            self.current_gate_index += 1
        else: # new action begin
            self.current_gate_index = 0
            self.chosen_sizes = []
            self.current_path_index += 1
            self.sizedCellList = []
            done = False
            vec_reward = self._get_tns_drc(self.inline)
            info = {}
            # new episode begin
            if self.current_path_index >= len(self.CriticalPaths):
                self.current_path_index = 0
                self.inline = False
        
        return self._get_obs(), vec_reward, done, False, info

    def _get_gate_sizes(self):
        gate_sizes = np.zeros(len(self.CriticalPaths[self.current_path_index].Cellname_to_Cell.keys()), dtype=np.int32)  # total number of gates on paths
        gate_sizes[:len(self.chosen_sizes)] = [self.action_space_list.index(val) + 1 for val in self.chosen_sizes] # plus 1 to avoid 0, since 0 means gate remain its original size
        self.gate_sizes[self.current_path_index, :len(gate_sizes)] = gate_sizes # update the gate sizes observation
        return gate_sizes
    
    def _get_sized_cellDict(self):
        sizedCell = {}
        for (cellname, celltype), chosedsize in zip(self.CriticalPaths[self.current_path_index].Cellname_to_Cell.items(), self._get_gate_sizes()):
            originalsize = celltype
            cellfootprint = self.timingLib[celltype].footprint # Cell Timing Lib Structure
            maxsize = len(self.footprint[cellfootprint]) # target cell list
            if chosedsize == 0:
                pass # 0 means remain the original size
            elif chosedsize <= maxsize:
                newsize = self.footprint[cellfootprint][chosedsize - 1]
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
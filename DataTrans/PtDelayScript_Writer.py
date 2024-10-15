import sys
import os
import pickle
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var
import PtCellRpt_Parser
# import DataBuilder

# Save_Path = Global_var.Saved_Data_Path

def Write_PtDelayScrip(design):
    print(f'Writing {design} PtDelay scripts.')
    inCellRpt = Global_var.PtRpt_Path + design + '_cell.rpt'
    outPtScript = Global_var.PtScript_Path + design + '_Delay.tcl'
    PtCells = PtCellRpt_Parser.Read_PtCellRpt(inCellRpt)
    # save_dir = Save_Path + 'Verilog/' + design
    # if not os.path.exists(save_dir):
    #     DataBuilder.BuildVerilog(design)
    # Verilog, VeriNets, VeriCells = DataBuilder.LoadVerilog(design)
    with open(outPtScript, 'w') as outfile:
        for cellname, cell in PtCells.items():
            for inpin in cell.inpins:
                for outpin in cell.outpins:
                    outfile.write('report_delay_calculation -from ' + cellname + '/' + inpin + ' -to ' + cellname + '/' + outpin + ' -nosplit\n')
    # save_dir = Save_Path + 'TimingArc'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save_path = os.path.join(save_dir, design + '_CellArc.sav')
    # with open(save_path, 'wb') as outfile:
    #     pickle.dump(CellArcDelay, outfile)

import os
import sys
Path = os.path.dirname(os.path.abspath(__file__))
parser_dir = os.path.join(Path, 'Parsers')
model_dir = os.path.join(Path, 'Model')
builder_dir = os.path.join(Path, 'DataTrans')
work_dir = os.path.join(Path, 'work')
RL_dir = os.path.join(Path, 'RL_Algorithm')
sys.path.append(parser_dir);sys.path.append(model_dir);sys.path.append(builder_dir);sys.path.append(work_dir);sys.path.append(RL_dir)

#path to the Library
# Lib_Path = '/home/md1/eda/techlibs/SMIC/SCC14NSFP_90SDB_9TC16_RVT_v1p0a/Liberty/scc14nsfp_90sdb_9tc16_rvt_ssg_v0p63_125c_ccs.lib'
Lib_Path = os.path.join(work_dir, 'Timing_Lib/scc14nsfp_90sdb_9tc16_rvt_ssg_v0p63_125c_ccs.lib')
# path to the Icc2 Rpt
Icc2Rpt_Path = os.path.join(work_dir, 'Icc2Rpt/')

# path to the def file
Def_Path = os.path.join(work_dir, 'Icc2Output/')

# path to the verilog file
Verilog_Path = os.path.join(work_dir, 'Icc2Output/')

# path to the timing Rpt
PtRpt_Path = os.path.join(work_dir, 'PtRpt/')

# path to write the PT scripts
PtScript_Path = os.path.join(work_dir, 'Delay_scripts/')

# Saved data path
Saved_Data_Path = os.path.join(Path, 'DataTrans/Data/')

# Transformed data path
Trans_Data_Path = os.path.join(Path, 'DataTrans/Processed_Data/')

# scales
scales = [4, 128, 512]
# designs 
# Designs = ['wb_conmax_top']
# Designs = ['aes_cipher_top', 'sasc_top', 'nova']
Designs = ['vga_enh_top', 'aes_cipher_top', 'ac97_top', 'des', 'ecg', 'eth_top', 'fpu', 'i2c_master_top', 'mc_top', 'nova', 'openGFX430', 'pci_bridge32', 'pcm_slv_top', 's38417', 'sasc_top', 'spi_top', 'tate_pairing', 'tv80_core', 'usbf_top', 'wb_conmax_top']
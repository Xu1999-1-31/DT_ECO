#!/bin/bash

# 定义设计名称列表
designs=("vga_enh_top" "aes_cipher_top" "ac97_top" "des" "ecg" "eth_top" "fpu" "i2c_master_top" "mc_top" "nova" "openGFX430" "pci_bridge32" "pcm_slv_top" "s38417" "sasc_top" "spi_top" "tate_pairing" "tv80_core" "usbf_top" "wb_conmax_top")

# 循环处理每个设计
for design in "${designs[@]}"
do
    echo "Processing design: $design"

    # 创建一个临时TCL脚本并将top_design设置为当前设计
    tcl_script="temp_$design.tcl"
    
    # 复制原始TCL脚本，并替换 top_design
    sed "s/set bench .*/set bench $design/" ../icc2_rpt.tcl > $tcl_script

    # 运行TCL脚本
    icc2_shell -f $tcl_script

    # 清理临时文件
    rm $tcl_script
done


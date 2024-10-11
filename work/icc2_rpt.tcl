set starttime [clock seconds]                                                                                                            
echo "INFORM: Start job at: " [clock format $starttime -gmt false]


set bench aes_cipher_top
open_lib ../Icc2Ndm/${bench}_nlib
open_block ${bench}
link_block

report_net -physical -significant_digits 6 -verbose > ../Icc2Rpt/${bench}_net.rpt

set bboxs [get_attribute -objects [get_cell -filter "(ref_name !~ *FILL*) && (ref_name !~ *ENDCAP*) && (ref_name !~ *DCAP*)"] -name boundary]

set cells [get_object_name [get_cell -filter "(ref_name !~ *FILL*) && (ref_name !~ *ENDCAP*) && (ref_name !~ *DCAP*)"]]

if {[file exists ../Icc2Rpt/${bench}_cell.rpt]} {
    file delete -force ../Icc2Rpt/${bench}_cell.rpt
}                                                                                                                                        

#set file_output [open ./outputs_rpt/${bench}_cell.rpt w]

foreach cell $cells {
    echo $cell >> ../Icc2Rpt/${bench}_cell.rpt
}


foreach bbox $bboxs {
    echo $bbox >> ../Icc2Rpt/${bench}_cell.rpt
}

source ../list_pin_bbox.tcl > ../Icc2Rpt/${bench}_pin.rpt
source ../list_port_bbox.tcl > ../Icc2Rpt/${bench}_port.rpt

source ../list_drc_errors.tcl > ../Icc2Rpt/${bench}_drc.rpt

set dimens 512
source ../report_congestion.tcl > ../Icc2Rpt/${bench}_congestion_${dimens}.rpt

write_verilog -exclude {cover_cells well_tap_cells filler_cells end_cap_cells corner_cells } ../Icc2Output/${bench}_route.v
write_sdc -output ../Icc2Output/${bench}_route.sdc
write_def  ../Icc2Output/${bench}_route.def

set_app_option -name extract.enable_coupling_cap -value true
set_parasitics_parameters -early_spec rcworst -late_spec rcbest
write_parasitics  -output ../Icc2Output/${bench} 

set endtime   [clock seconds]
echo "INFORM: End job at: " [clock format $endtime -gmt false]
set pwd [pwd]
set runtime "[format %02d [expr ($endtime - $starttime)/3600]]:[format %02d [expr (($endtime - $starttime)%3600)/60]]:[format %02d [expr ((($endtime - $starttime))%3600)%60]]"
echo [format "%-15s %-2s %-70s" " | runtime" "|" "$runtime"]


exit
report_net -physical -significant_digits 6 -verbose > ../Icc2Rpt/${bench}_eco_net.rpt

set bboxs [get_attribute -objects [get_cell -filter "(ref_name !~ *FILL*) && (ref_name !~ *ENDCAP*) && (ref_name !~ *DCAP*)"] -name boundary]

set cells [get_object_name [get_cell -filter "(ref_name !~ *FILL*) && (ref_name !~ *ENDCAP*) && (ref_name !~ *DCAP*)"]]

if {[file exists ../Icc2Rpt/${bench}_eco_cell.rpt]} {
    file delete -force ../Icc2Rpt/${bench}_eco_cell.rpt
}                                                                                                                                        

#set file_output [open ./outputs_rpt/${bench}_eco_cell.rpt w]

foreach cell $cells {
    echo $cell >> ../Icc2Rpt/${bench}_eco_cell.rpt
}


foreach bbox $bboxs {
    echo $bbox >> ../Icc2Rpt/${bench}_eco_cell.rpt
}

source ../list_pin_bbox.tcl > ../Icc2Rpt/${bench}_eco_pin.rpt
source ../list_port_bbox.tcl > ../Icc2Rpt/${bench}_eco_port.rpt

source ../list_drc_errors.tcl > ../Icc2Rpt/${bench}_eco_drc.rpt

set dimens 512
source ../report_congestion.tcl > ../Icc2Rpt/${bench}_eco_congestion_${dimens}.rpt

write_verilog -exclude {cover_cells well_tap_cells filler_cells end_cap_cells corner_cells } ../Icc2Output/${bench}_eco_route.v
write_sdc -output ../Icc2Output/${bench}_eco_route.sdc
write_def  ../Icc2Output/${bench}_eco_route.def

set_app_option -name extract.enable_coupling_cap -value true
set_parasitics_parameters -early_spec rcbest -late_spec rcworst
write_parasitics  -output ../Icc2Output/${bench}_eco
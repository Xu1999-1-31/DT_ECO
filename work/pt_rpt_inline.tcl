set starttime [clock seconds]
echo "INFORM: Start job at: " [clock format $starttime -gmt false]
set is_si_enabled false

set top_design aes_cipher_top

set link_library "* ../Timing_Lib/scc14nsfp_90sdb_9tc16_rvt_ssg_v0p63_125c_ccs.db"


set netlist "../VerilogInline/${top_design}_route.v"
set sdc "../Icc2Output/${top_design}_route.sdc"
set spef "../Icc2Output/${top_design}.rcworst_125_1.08_1.08_1_1.spef"

source -e -v ../pt_variable.tcl

set NET_FILE $netlist 
set SDC_FILE $sdc
set SPEF_FILE $spef


set_app_var read_parasitics_load_locations true
set_app_var eco_allow_filler_cells_as_open_sites true
###################################################################
read_verilog  $NET_FILE
link


read_parasitics -keep_capacitive_coupling  -format SPEF  $SPEF_FILE

source -e -v $SDC_FILE

set_propagated_clock [all_clocks]

set timing_remove_clock_reconvergence_pessimism true
     
set timing_disable_clock_gating_checks true  
set timing_report_unconstrained_paths true


update_timing -full

report_timing -nosplit -nets -input_pins -transition_time -capacitance -significant_digit 6 -max_path 100000 > ../PtRpt/${top_design}_inline.rpt
report_global_timing -significant_digits 8 > ../PtRpt/${top_design}_inline_global.rpt
report_cell -connections -nosplit > ../PtRpt/${top_design}_inline_cell.rpt
# report_net -connections -verbose > ../PtRpt/${top_design}_inline_net.rpt
# report_analysis_coverage -status_details violated -check_type setup -nosplit -significant_digits 8 > ../PtRpt/${top_design}_inline_violated_endpoint.rpt
# report_analysis_coverage -status_details met -check_type setup -nosplit -significant_digits 8 > ../PtRpt/${top_design}_inline_met_endpoint.rpt
# report_analysis_coverage -status_details untested -check_type setup -nosplit -significant_digits 8 > ../PtRpt/${top_design}_inline_untested_endpoint.rpt
# write_sdf -significant_digits 8 -input_port_nets -output_port_nets ../PtRpt/${top_design}_inline.sdf


if {[file exists ../Delay_scripts/${top_design}_Delay.tcl]} {
    source ../Delay_scripts/${top_design}_Delay.tcl > ../PtRpt/${top_design}_inline_Delay.rpt
}

set endtime   [clock seconds]
echo "INFORM: End job at: " [clock format $endtime -gmt false]
set pwd [pwd]
set runtime "[format %02d [expr ($endtime - $starttime)/3600]]:[format %02d [expr (($endtime - $starttime)%3600)/60]]:[format %02d [expr ((($endtime - $starttime))%3600)%60]]"
echo [format "%-15s %-2s %-70s" " | runtime" "|" "$runtime"]
exit

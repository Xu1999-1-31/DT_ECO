set sh_message_limit 0
set sdc_save_source_file_information            true

# Timing variable
set svr_keep_unconnected_nets                   true
set timing_save_pin_arrival_and_slack           true
set timing_report_unconstrained_paths           true

set timing_enable_max_capacitance_set_case_analysis true
# CPRP
set timing_remove_clock_reconvergence_pessimism true
set timing_crpr_threshold_ps                    5.0
#set timing_clock_reconvergence_pessimism        normal
#set timing_input_port_default_clock             false

# Wireload
#set auto_wire_load_selection false

# To prevent the gating signal propagating into the clock
#set timing_clock_gating_propagate_enable        true

# SI
if {$is_si_enabled} {
  set si_enable_analysis                    true
  set si_xtalk_double_switching_mode        clock_network
  set si_xtalk_analysis_effort_level        high
#  set si_xtalk_reselect_delta_delay         0.01
#  set si_xtalk_reselect_delta_delay_ratio   0.95
#  set si_xtalk_reselect_max_mode_slack      0
#  set si_xtalk_reselect_min_mode_slack      0
#  set si_xtalk_reselect_clock_network       true
  set si_analysis_logical_correlation_mode  false
  set si_xtalk_exit_on_max_iteration_count  3
}

########for USB, usb tsmc65lp25databook 208###############
#set access_internal_pins true

set pins [get_pins -hier -filter "is_hierarchical==false && full_name!~*/VDD && full_name!~*/VSS && full_name!~*/VNW && full_name!~*/VPW && full_name!~*FILL* && full_name!~*ENDCAP* && full_name!~*DCAP*"]
foreach_in_collection pin $pins {
	set pin_name [get_attribute $pin full_name]
	set bbox [get_attribute $pin bbox]
	puts "$pin_name $bbox"
}

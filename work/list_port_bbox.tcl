set ports [get_ports]
foreach_in_collection port $ports {
	set port_name [get_attribute $port full_name]
	set bbox [get_attribute $port bbox]
	puts "$port_name $bbox"
}

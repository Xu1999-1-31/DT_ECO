 #==========================================================================================================
 #
 #		                              ******************                               
 #		                           *************************                           
 #		                       ********              ********                          
 #		                           ****                ********                        
 #		                    * *       *                *********                       
 #		                     *                         *********                       
 #		                    * *         *              **********                      
 #		                             ****              **********                      
 #		                          *******             ***********                      
 #		                       **********            *************                     
 #		                           ******          **************                      
 #		                           ******         ***************                      
 #		                          *******        **************                        
 #		                         ********       *************                          
 #		                        *********                                              
 #		                       **********                                              
 #		                      ***********                       *                      
 #		                     ************ ******                **                     
 #		                    ************* ************          ***                    
 #		                    ************* *****************     ****                   
 #		                   ************** ***************************                  
 #		                  *************** ****************************                 
 #		                 **************** *****************************                
 #		                                                                               
 #		HH    HH   HH          HHHHHHH     HHHHHHHHH   HHHHHHHH     HHHHH     HH    HH 
 #		HH    HH   HH          HH    HHH      HH       HH         HH    HHH   HH    HH 
 #		HH    HH   HH          HH    HHH      HH       HH         HH          HH    HH 
 #		HH    HH   HH          HHHHHHHH       HH       HHHHHHHH   HH          HHHHHHHH 
 #		HH    HH   HH          HH             HH       HH         HH          HH    HH 
 #		HH    HH   HH          HH             HH       HH         HH     HH   HH    HH 
 #		 HHHHHH    HHHHHHHHH   HH             HH       HHHHHHHH     HHHHH     HH    HH 
 #==========================================================================================================
 #
 #	Nanjing Low Power IC Technology Institute Co.,Ltd
 #	17 Xinghuo Road, Jiangbei New District
 #	Nanjing, Jiangsu Province, China
 #
 #==========================================================================================================
 #
 #			STATEMENT OF USE AND CONFIDENTIALITY
 #	This information contains confidential and proprietary information of ULPTECH.
 #	No part of this information may be reproduced, transmitted, transcribed, stored
 #	in a retrieval system, or translated into any human or computer language, in any
 #	form or by any means, electronic, mechanical, magnetic, optica, chemical, manual,
 #	or otherwise, without the prior written permission of ULPTECH. This information
 #	was prepared for informational purpose and is for use by ULPTECH's customers only.
 #	ULPTECH reserves the right to make changes in the information at any time and
 #	without notice.
 #
 #==========================================================================================================
 #
 #	Author          :	Richard Ren
 #	Email           :	renlz@ulptech.com
 #	File created    :	2024-08-21 09:55
 #	Last modified   :	2024-08-21 11:20
 #	Filename        :	report_congestion.tcl
 #	Description     :
 #
 #==========================================================================================================
#set dimens_init 4
#for {set scale_ratio 0} {$scale_ratio < 11} {incr scale_ratio} {
#	set dimens [expr $dimens_init/(2**$scale_ratio)]
	set bbox_block [get_attribute [current_block] boundary_bbox]
	set ll [lindex $bbox_block 0]
	set ur [lindex $bbox_block 1]
	set llx_block [lindex $ll 0]
	set lly_block [lindex $ll 1]
	set urx_block [lindex $ur 0]
	set ury_block [lindex $ur 1]

	for {set i 0} {$i < $dimens} {incr i} {
		set llx [expr $llx_block+($urx_block-$llx_block)/$dimens*$i]
		set urx [expr $urx_block+($urx_block-$llx_block)/$dimens*($i+1)]
		for {set j 0} {$j < $dimens} {incr j} {
			set lly [expr $lly_block+($ury_block-$lly_block)/$dimens*$j]
			set ury [expr $ury_block+($ury_block-$lly_block)/$dimens*($j+1)]
			echo "## X/Y tile count $dimens*$dimens; row=$i, column=$j; bbox (($llx, $lly), ($urx, $ury))"
			eval "report_congestion -boundary {{$llx $lly} {$urx $ury}} -layers * -nosplit"
		}
	}
#}

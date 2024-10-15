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
 #	File created    :	2024-08-21 14:59
 #	Last modified   :	2024-08-21 16:01
 #	Filename        :	list_drc_errors.tcl
 #	Description     :
 #
 #==========================================================================================================
check_design -checks routes
open_drc_error_data zroute.err
set drc_errs [get_drc_errors -error_data zroute.err]
set index 1
foreach_in_collection drc_err $drc_errs {
	set err_type [get_attribute $drc_err error_type]
	set polygons [get_attribute $drc_err polygons]
	set layer [get_attribute $drc_err layers]
	echo "#$index --> $layer $err_type $polygons"
	incr index
}



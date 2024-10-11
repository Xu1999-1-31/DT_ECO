import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var
import PtDelayScript_Writer

def ReBuildPtScripts(design=None):
    if design:
        PtDelayScript_Writer.Write_PtDelayScrip(design)
    else:
        for design in Global_var.Designs:
            PtDelayScript_Writer.Write_PtDelayScrip(design)

# ReBuildPtScripts()
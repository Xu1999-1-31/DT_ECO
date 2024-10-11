import MultiCoreAccelerator
import DataBuilder
import PhysicalDataTrans
import TimingGraphTrans
import Global_var

def ReBuildAllData(design=None):
    DataBuilder.BuildTimingLib()
    if design:
        MultiCoreAccelerator.BuildAllData(design)
    else:
        MultiCoreAccelerator.MultiCoreAccelerator()

def ReBuildProcessedData(design=None, scales=[512]):
    if design:
        TimingGraphTrans.TimingGraphTrans(design)
        for scale in scales:
            PhysicalDataTrans.PhysicalDataTrans(design, scale)
    else:
        for design in Global_var.Designs:
            TimingGraphTrans.TimingGraphTrans(design)
            for scale in scales:
                PhysicalDataTrans.PhysicalDataTrans(design, scale)

ReBuildProcessedData()
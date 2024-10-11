import DataBuilder
import TimingGraphTrans
import PhysicalDataTrans
import MergeMultiPath
import os
from PIL import Image

# DataBuilder.BuildCellLayout()
# DataBuilder.LoadCellLayout('aes_cipher_top')
# DataBuilder.BuildMetalLayer()
# DataBuilder.LoadMetalLayer('aes_cipher_top')
# DataBuilder.BuildRouteCongestion()
# DataBuilder.LoadRouteCongestion('aes_cipher_top')
# DataBuilder.BuildCellDensity()
# DataBuilder.LoadCellDensity('aes_cipher_top')
# DataBuilder.BuildDrcMap()
# DataBuilder.LoadDrcMap('sasc_top')
# DataBuilder.BuildTimingLib()
# DataBuilder.LoadTimingLib()
# DataBuilder.LoadNormalizedTimingLib()
# DataBuilder.BuildVerilog('nova')
# DataBuilder.LoadVerilog('sasc_top')
# DataBuilder.BuildPtRpt('aes_cipher_top')
# DataBuilder.LoadPtRpt('aes_cipher_top')
# DataBuilder.BuildTimingArc('aes_cipher_top')
# DataBuilder.BuildEndPoint('aes_cipher_top')
# DataBuilder.LoadEndPoint('aes_cipher_top')
# TimingGraphTrans.TimingGraphTrans('aes_cipher_top')
# DataBuilder.BuildPortData('aes_cipher_top')
# PhysicalDataTrans.PhysicalDataTrans('vga_enh_top', 512)
# DataBuilder.BuildGlobalTimingData('vga_enh_top')
# DataBuilder.BuildPinLayout('aes_cipher_top')
# MergeMultiPath.MergeMultiPath('aes_cipher_top')
# TimingGraphTrans.LoadTimingGraph('aes_cipher_top_eco', True)
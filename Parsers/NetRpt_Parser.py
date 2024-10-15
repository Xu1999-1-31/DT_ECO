class Segment:
    def __init__(self, metal, width, startX, startY, endX, endY):
        self.metal = metal
        self.width = width
        self.sx = startX
        self.sy = startY
        self.ex = endX
        self.ey = endY
    def __repr__(self):
        return (f'Segment(metal={self.metal}, width={self.width}, '
                f'start=({self.sx:.4f}, {self.sy:.4f}), end=({self.ex:.4f}, {self.ey:.4f}))')

class Net:
    def __init__(self, name):
        self.name = name
        self.segs = []
    def __repr__(self):
        segs_repr = ',\n  '.join([repr(seg) for seg in self.segs])
        return (f'Net(name={self.name}, '
                f'segments=[\n  {segs_repr}\n])')

def Read_NetRpt(inrpt):
    with open(inrpt, 'r') as infile:
        rptNets = []
        lines = infile.readlines()
        for line in lines:
            index = line.split()
            if(len(index) > 1):
                if(index[0] == 'flat_net'):
                    if 'newNet' in locals():
                        rptNets.append(newNet)
                    newNet = Net(index[1])
                if('M' in index[0] and len(index[0]) == 2):
                    newSeg = Segment(index[0], float(index[1]), float(index[2]), float(index[3]), float(index[4]), float(index[5]))
                    newNet.segs.append(newSeg)
        if 'newNet' in locals():
            rptNets.append(newNet)
    return rptNets
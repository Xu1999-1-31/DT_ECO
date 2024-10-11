import re

class componement:
    def __init__(self, name, type, x, y, position):
        self.name = name
        self.type = type
        self.x = x
        self.y = y
        self.position = position
    def __repr__(self):
        return (f'Cell(name={self.name}, '
                f'type={self.type}, '
                f'x={self.x}, '
                f'y={self.y}, '
                f'pos={self.position})\n')

class Segment:
    def __init__(self, metal, startX, startY, endX, endY):
        self.metal = metal
        self.sx = startX
        self.sy = startY
        self.ex = endX
        self.ey = endY
    def __repr__(self):
        return (f'Segment(metal={self.metal}, '
                f'start=({self.sx:.4f}, {self.sy:.4f}), end=({self.ex:.4f}, {self.ey:.4f}))')

class Net:
    def __init__(self, name):
        self.name = name
        self.pins = []
        self.segs = []
    def __repr__(self):
        pins_repr = ', '.join(self.pins)
        segs_repr = ',\n  '.join([repr(seg) for seg in self.segs])
        return (f'Net(name={self.name}, '
                f'pins=[{pins_repr}], '
                f'segments=[\n  {segs_repr}\n])')


def Read_def(indef):
    scaler = 1000
    coreArea = []
    components = []
    nets = []
    layers = []
    com_flag = 0
    net_flag = 0
    routeflag = 0
    with open(indef, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            index = line.split()
            if(len(index) > 1):
                # die area
                if(index[0] == 'DIEAREA'):
                    for i in range(len(index) - 2):
                        if(index[i] == "("):
                            coreArea.append((float(index[i+1])/scaler, float(index[i+2])/scaler))
                # Layers
                if(index[0] == 'TRACKS'):
                    if index[8] not in layers and len(index[8]) == 2:
                        layers.append(index[8])
                # components
                if(index[0] == 'END' and index[1] == 'COMPONENTS'):
                    com_flag = 0
                if(com_flag == 1):
                    # remove Filler
                    if('FILL' in index[2] or 'BOUNDARY' in index[2] or 'TAP' in index[2] or 'ENDCAP' in index[2]):
                        pass
                    else:
                        new_com = componement(index[1], index[2], float(index[6])/scaler, float(index[7])/scaler, index[9])
                        components.append(new_com)
                if(index[0] == 'COMPONENTS'):
                    com_flag = 1
                # nets
                if(index[0] == "END" and index[1] == "NETS"):
                    net_flag = 0
                if(net_flag == 1):
                    # net name
                    if(index[0] == '-'):
                        newNet = Net(index[1])
                    # net pins
                    if(index[0] == '('):
                        newNet.pins.append(index[1])
                    # net routing part
                    if(index[0] == '+' and index[1] == 'USE'):
                        routeflag = 0
                        nets.append(newNet)
                    if(index[0] == '+' and index[1] == 'ROUTED'):
                        routeflag = 1
                    if(routeflag == 1):
                        matches = re.findall(r'\(\s*([^)]*?)\s*\)', line)
                        matches = [match.strip() for match in matches]
                        if(len(matches) == 1):
                            pass
                        else:
                            if(index[0] == '+'):
                                metal = index[2]
                            else:
                                metal = index[1]
                            coordinates = []
                            for match in matches:
                                coords = match.split()
                                if len(coords) == 2 or len(coords) == 3:
                                    if coords[0] == '*':
                                        coords[0] = coordinates[-1][0]
                                    if coords[1] == '*':
                                        coords[1] = coordinates[-1][1]
                                    coordinates.append((float(coords[0])/scaler, float(coords[1])/scaler))
                            for i in range(len(coordinates) -1):
                                newSeg = Segment(metal, coordinates[i][0], coordinates[i][1], coordinates[i+1][0], coordinates[i+1][1])
                                newNet.segs.append(newSeg)
                if(index[0] == "NETS"):
                    net_flag = 1
    return coreArea, components, nets, layers
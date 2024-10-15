class Cell:
    def __init__(self, name, pin1, pin2, pin3, pin4):
        self.name = name
        self.pin1 = pin1
        self.pin2 = pin2
        self.pin3 = pin3
        self.pin4 = pin4
    def __repr__(self):
        return (f'Cell(name={self.name}, pin1=({self.pin1[0]:.4f}, {self.pin1[1]:.4f}), '
                f'pin2=({self.pin2[0]:.4f}, {self.pin2[1]:.4f})), pin3=({self.pin3[0]:.4f}, {self.pin3[1]:.4f})), '
                f'pin4=({self.pin4[0]:.4f}, {self.pin4[1]:.4f}))')

def Read_CellRpt(inrpt):
    with open(inrpt, 'r') as infile:
        lines = infile.readlines()
        cellname = []
        cellcount = 0
        cellList = []
        for line in lines:
            index = line.split()
            if(len(index) > 0):
                if('{' not in index[0]):
                    cellname.append(index[0])
                else:
                    newcell = Cell(cellname[cellcount], (float(index[0].replace('{', '')), \
                                    float(index[1].replace('}', ''))), (float(index[2].replace('{', '')), \
                                    float(index[3].replace('}', ''))), (float(index[4].replace('{', '')), \
                                    float(index[5].replace('}', ''))), (float(index[6].replace('{', '')), \
                                    float(index[7].replace('}', ''))))
                    cellList.append(newcell)
                    cellcount += 1
    
    return cellList

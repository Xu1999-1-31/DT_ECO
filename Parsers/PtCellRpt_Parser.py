import linecache

class Cell:
    def __init__(self):
        self.type = ''
        self.name = ''
        self.inpins = []
        self.outpins = []
    def __repr__(self):
        inpins_repr = ', '.join([f'{pin}' for pin in self.inpins])
        outpins_repr = ', '.join([f'{pin}' for pin in self.outpins])
        return f"Cell(type='{self.type}', name='{self.name}', \ninpins={{ {inpins_repr} }}, \noutpins={{ {outpins_repr} }})"

def Read_PtCellRpt(inrpt):
    cells = {}
    with open(inrpt, 'r') as infile:
        linecount = 0
        for line in infile:
            linecount += 1
            if line.startswith('Connections'):
                try:
                    newcell
                except NameError:
                    pass
                else:
                    cells[newcell.name] = newcell
                index = line.split()
                newcell = Cell()
                newcell.name = index[3].replace('\'', '').replace(':', '')
            index = line.split()
            if(len(index) > 1):
                if index[0] == 'Reference:':
                    newcell.type = index[1]
                elif index[0] == 'Input' and index[1] == 'Pins':
                    i = 1
                    while True:
                        newline = linecache.getline(inrpt, linecount+i)
                        index = newline.split()
                        if len(index) == 0:
                            break
                        elif '---' not in index[0]:
                            newcell.inpins.append(index[0])
                        i += 1
                elif index[0] == 'Output' and index[1] == 'Pins':
                    i = 1
                    while True:
                        newline = linecache.getline(inrpt, linecount+i)
                        index = newline.split()
                        if len(index) == 0:
                            break
                        elif '---' not in index[0]:
                            newcell.outpins.append(index[0])
                        i += 1
        cells[newcell.name] = newcell
    return cells
                
class Drc:
    def __init__(self, num):
        self.number = num
        self.pins = []
    def __repr__(self):
        pin_str = ', '.join([str(row) for row in self.pins])
        return f'Drc(num=#{self.number}, pins=[{pin_str}])\n'

def Read_Drc(inrpt):
    with open(inrpt, 'r') as infile:
        lines = infile.readlines()
        DrcList = []
        for line in lines:
            if line.startswith('#'):
                index = line.split()
                newDrc = Drc(index[0].replace('#', ''))
                for i in range(len(index)):
                    if '{' in index[i]:
                        newDrc.pins.append((float(index[i].replace('{', '')), float(index[i+1].replace('}', ''))))
                DrcList.append(newDrc)
    
    return DrcList

class Cell:
    def __init__(self):
        self.type = ''
        self.name = ''
        self.pins = {} # pin -> net
    def __repr__(self):
        pins_repr = ', '.join([f'{pin}: {net}' for pin, net in self.pins.items()])
        return f"Cell(type='{self.type}', name='{self.name}', pins={{ {pins_repr} }})"

class Verilog:
    def __init__(self):
        self.name = ''
        self.ports = []
        self.inputs = []
        self.outputs = []
        self.wires = []
        self.cells = []
    def __repr__(self):
        cells_repr = ', \n'.join([repr(cell) for cell in self.cells])
        return (f"Verilog(name='{self.name}', \nports={self.ports}, "
                f"\ninputs={self.inputs}, \noutputs={self.outputs}, "
                f"\nwires={self.wires}, \ncells=[{cells_repr}])")

def generate_split_pin(range_str, base_str):
    range_str = range_str.strip('[]')
    x_str, y_str = range_str.split(':')
    
    x = int(x_str)
    y = int(y_str)
    
    start = min(x, y)
    end = max(x, y)
    
    return [f"{base_str}[{i}]" for i in range(start, end + 1)]

def pin_reader(indata, outlist):
    for i in range(len(indata)):
        if '[' in indata[i] and ':' in indata[i]:
            pins = generate_split_pin(indata[i], indata[i+1])
            for pin in pins:
                outlist.append(pin)
        elif '[' in indata[i-1]:
            pass
        else:
            outlist.append(indata[i])

def Read_Verilog(inVerilog):
    with open(inVerilog, 'r') as infile:
        content = infile.read().replace('\n', ' ')
    content = content.split()
    newVerilog = Verilog()
    for i in range(len(content)):
        if i < len(content) -2:
            if content[i] == 'module' and content[i+2] == '(':
                module = []
                j = i
                while content[j] != ';':
                   module.append(content[j])
                   j += 1
                newVerilog.name = content[1]
                for j in range(3, len(module) -1):
                    if module[j] != ',':
                        newVerilog.ports.append(module[j])
        if content[i] == 'input':
            j = i + 1
            input = []
            while content[j] != ';':
                if(content[j] != ','):
                    input.append(content[j])
                j += 1
            pin_reader(input, newVerilog.inputs)
        if content[i] == 'output':
            j = i + 1
            output = []
            while content[j] != ';':
                if(content[j] != ','):
                    output.append(content[j])
                j += 1
            pin_reader(output, newVerilog.outputs)
        if content[i] == 'wire':
            j = i + 1
            wire = []
            while content[j] != ';':
                if(content[j] != ','):
                    wire.append(content[j])
                j += 1
            pin_reader(wire, newVerilog.wires)
        if '90S9T16R' in content[i]:
            j = i
            cell = []
            while ';' not in content[j]:
                cell.append(content[j])
                j += 1
            newCell = Cell()
            newCell.name = cell[1]
            newCell.type = cell[0]
            for j in range(len(cell)):
                if '.' in cell[j] and cell[j] != '.VDD' and cell[j] != '.VSS' and cell[j] != '.VNW' and cell[j] != '.VPW':
                    newCell.pins[cell[j].replace('.', '')] = cell[j+2]
            newVerilog.cells.append(newCell)
    # print(newVerilog)
    return newVerilog

    # print(newVerilog.wires)

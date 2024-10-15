import re

class Path:
    def __init__(self):
        self.Cellarcs = []
        self.Netarcs = []
        self.Pins = []
        self.Cells = []
        self.setup = 0
        self.required_time = 0
        self.slack = 0
        self.pathdelay = 0
        self.Temp = 0
        self.clk = None
        self.Startpoint = ''
        self.Endpoint = ''
        self.constrain = ''
    def __repr__(self):
        cellarcs_repr = ',\n  '.join([repr(arc) for arc in self.Cellarcs])
        netarcs_repr = ',\n  '.join([repr(arc) for arc in self.Netarcs])
        pins_repr = ','.join([repr(pin) for pin in self.Pins])
        cells_repr = ','.join(self.Cells)
        return (f'<Path rpt:\nStartpoint: {self.Startpoint} Endpoint: {self.Endpoint}\npathdelay: {self.pathdelay} slack: {self.slack} setup: {self.setup} required_time: {self.required_time}\nconstrain: {self.constrain}\nclk: {self.clk}\n'
                f'pins:[{pins_repr}]\ncells:[{cells_repr}]\ncellarcs:[\n  {cellarcs_repr}\n]\nnetarcs:[\n  {netarcs_repr}\n]>')

class Net:
    def __init__(self):
        self.name = ''
        self.fanout = 0
        self.cap = 0
    def __repr__(self):
        return f"<Net Name: {self.name}, Fanout: {self.fanout}, Cap: {self.cap}>"

class Constrain:
    def __init__(self):
        self.name = ''
        self.cell = ''
        self.rf =''
    def __repr__(self):
        return f"<Constrain Name: {self.name}, Cell: {self.cell}, rf: {self.rf}>"

class Clk:
    def __init__(self):
        self.name = ''
        self.delay_start = 0
        self.delay_end = 0
        self.start_edge = 0 # rise edge 0, fall edge 0.05
        self.end_edge = 0 # rise edge 0.1 fall edge 0.15
        self.T = 0
    def __repr__(self):
        return f"<CLK Name: {self.name}, Delay_start: {self.delay_start}, Delay_end: {self.delay_end}, Start_edge: {self.start_edge}, End_edge: {self.end_edge}>"

class Pin_and_Arc_base:
    def __init__(self):
        self.name = ''
        self.outtrans = 0
        self.delay = 0
        self.rf = '' # rise or fall
    def __repr__(self):
        return f"<Name: {self.name}, outTrans: {self.outtrans}, Delay: {self.delay}, rf : {self.rf}>"

class rpt_Pin(Pin_and_Arc_base):
    def __init__(self):
        super().__init__()  
        self.map = ''  
    def __repr__(self):
        return f"<Name: {self.name}, outTrans {self.outtrans}, Map: {self.map}, Delay: {self.delay}, rf : {self.rf}>"

class Cell_arc(Pin_and_Arc_base):
    def __init__(self):
        super().__init__()  
        self.intrans = '' 
        self.cell = '' 
    def __repr__(self):
        return f"<Cell_arc: {self.name}, Cell: {self.cell}, inTrans: {self.intrans}, outTrans: {self.outtrans}, Delay: {self.delay}, rf: {self.rf}>"

    def __eq__(self, other):
        if not isinstance(other, Cell_arc):
            return NotImplemented
        return (self.name == other.name and
                self.cell == other.cell and
                self.outtrans == other.outtrans and
                self.delay == other.delay and
                self.rf == other.rf)

    def __hash__(self):
        # hash key based on name, cell, outtrans, delay, rf
        return hash((self.name, self.cell, self.outtrans, self.delay, self.rf))


class Net_arc(Pin_and_Arc_base):
    def __init__(self):
        super().__init__()  
        self.net_name = ''
        self.fanout = 0
        self.cap = 0  
    def __repr__(self):
        return f"<Net_arc: {self.name}, Net: {self.net_name}, outTrans: {self.outtrans}, Delay: {self.delay}, rf: {self.rf}, Fanout: {self.fanout}, Cap: {self.cap}>"

    def __eq__(self, other):
        if not isinstance(other, Net_arc):
            return NotImplemented
        return (self.name == other.name and
                self.net_name == other.net_name and
                self.outtrans == other.outtrans and
                self.delay == other.delay and
                self.rf == other.rf)

    def __hash__(self):
        # hash key based on name, net_name, outtrans, delay, rf
        return hash((self.name, self.net_name, self.outtrans, self.delay, self.rf))
    
def Read_PtRpt(inrpt):
    with open(inrpt, 'r') as infile:
        content = infile.read()
    pattern = re.compile(r'Startpoint(.*?)slack \(', re.DOTALL)
    timing_matches = pattern.findall(content)
    paths = []
    for timing_list in timing_matches:
        newpath = Path()
        timing_list = timing_list.replace('\n', '')
        timing_list = timing_list.replace('&', '')
        index = timing_list.find('data arrival time')
        # report before data arrive time // tcombine + clkdelay
        part1 = timing_list[:index] if index != -1 else timing_list
        # report after data arrive time // tsetup + T + clkdelay
        part2 = timing_list[index:-1] if index != -1 else timing_list
        index1 = part1.split()
        index2 = part2.split()
        # item in ()
        # part2 
        setup = 0
        required_time = 0
        clk = Clk()
        constrain = None #if the path is constrained
        for i in range(len(index2)):
            if(index2[i] == 'library'):
                if(index2[i+1] == 'setup' and index2[i+2] == 'time'):
                    setup = float(index2[i+3])
            if(index2[i].find('(rise') != -1 or index2[i].find('(fall') != -1):
                clk.name = index2[i-1]
                clk.end_edge = float(index2[i+2])
            if(index2[i].find('(propagated)') != -1):
                clk.delay_end = float(index2[i+1])
            if(index2[i].find('required') != -1 and index2[i+1].find('time') != -1):
                required_time = float(index2[i+2])
            if(index2[i].find('/') != -1 and index2[i+1].find('(') != -1):
                if(index2[i+3] == 'r' or index2[i+3] == 'f'):
                    constrain = Constrain()
                    constrain.name = index2[i]
                    constrain.cell = index2[i+1].replace('(', '').replace(')', '')
                    constrain.rf = index2[i+3]
                    

        # part1
        cells = []
        cell_count = 0
        for i in range(1, len(index1)):
            if(index1[i-1].find('/') != -1 and index1[i].find('(') != -1 and index1[i].find(')') != -1):
                if(cell_count%2 != 1):
                    cells.append(index1[i].replace('(', '').replace(')', ''))
                cell_count += 1
            if(index1[i].find('(propagated)') != -1):
                clk.delay_start = float(index1[i+1])
            if(index1[i].find('(rise') != -1 or index1[i].find('(fall') != -1):
                clk.start_edge = float(index1[i+2])
        clk.T = clk.end_edge - clk.start_edge
#                cell_count = 0        
#                for item in path_item:
#                    if(item.find('ris') == -1 and item.find('fall') == -1 and item.find('propagate') == -1 and item.find('VIOLATED') == -1 and item.find('input') == -1 and item.find('output') == -1 and item.find('net') == -1 and item.find('out') == -1 and item.find('in') == -1):
#                        if(cell_count%2 != 1):
#                            cells.append(item.replace('(', '').replace(')', ''))
#                        cell_count += 1
        pins = []
        nets = []
        path_delay = 0
        for i in range(len(index1)):
            if(index1[i].find('(') != -1):
                item = index1[i].replace('(', '').replace(')','')
                if(item in cells or item == 'out' or item == 'in'):
                    newpin = rpt_Pin()
                    newpin.name = index1[i-1]
                    newpin.map = item
                    newpin.outtrans = float(index1[i+1])
                    newpin.delay = float(index1[i+2])
                    newpin.rf = index1[i+4]
                    pins.append(newpin)
                if(item == 'net'):
                    newnet = Net()
                    newnet.name = index1[i-1]
                    newnet.fanout = float(index1[i+1])
                    newnet.cap = float(index1[i+2])
                    nets.append(newnet)


        path_delay = float(index1[-2])
        cell_arcs = []
        net_arcs = []
        # print(len(cells), len(pins))
        # remove last output DQ flip
        cell_arc_names = []
        for i in range(len(pins)):
            if(pins[i].name.find('/') != -1 and i != 0):
                pin_split1 = pins[i].name.split('/')
                pin_split2 = pins[i-1].name.split('/')
                if(pin_split1[0] == pin_split2[0] and pin_split1[1] != pin_split2[1]): # U22/I U22/ZN
                    newarc = Cell_arc()
                    newarc.name = pins[i-1].name + '->' + pins[i].name
                    cell_arc_names.append(newarc.name)
                    newarc.delay = pins[i].delay
                    newarc.outtrans = pins[i].outtrans
                    newarc.intrans = pins[i-1].outtrans
                    newarc.rf = pins[i].rf
                    newarc.cell = pins[i].map
                    cell_arcs.append(newarc)
        for i in range(1, len(pins)):
            arc_name = pins[i-1].name + '->' + pins[i].name
            if(arc_name not in cell_arc_names):  # not cellarc, then net arc
                newarc = Net_arc()
                newarc.name = arc_name
                newarc.delay = pins[i].delay
                newarc.outtrans = pins[i].outtrans
                newarc.rf = pins[i].rf
                net_arcs.append(newarc)
        for net_arc, net in zip(net_arcs, nets):
            net_arc.net_name = net.name
            net_arc.fanout = net.fanout
            net_arc.cap = net.cap
        newpath.Cellarcs = cell_arcs
        newpath.Netarcs = net_arcs
        newpath.Pins = pins
        newpath.Cells = cells
        newpath.setup = setup
        newpath.clk = clk
        newpath.required_time = required_time
        newpath.pathdelay = path_delay
        newpath.constrain = constrain
        paths.append(newpath)
    pattern = re.compile(r'Startpoint\: (.*?) \(', re.DOTALL)
    startpoint = pattern.findall(content)
    pattern = re.compile(r'Endpoint\: (.*?) \(', re.DOTALL)
    endpoint = pattern.findall(content)
    pattern = re.compile(r'.*slack \(.*')
    slack_line = pattern.findall(content)
    slack = []
    for line in slack_line:
        index = line.split()
        slack.append(float(index[-1]))
    if(len(startpoint) != len(slack) or len(startpoint) != len(endpoint) or len(startpoint) != len(paths)):
        raise ValueError("In design {design}: Lengths of startpoint, slack, endpoint, and paths are not equal.")
    for i in range(len(startpoint)):
        paths[i].Startpoint = startpoint[i].replace(' ','')
        paths[i].Endpoint = endpoint[i].replace(' ','')
        paths[i].slack = slack[i]
    
    return paths
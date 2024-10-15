class CellArc:
    def __init__(self):
        self.from_pin = ''
        self.to_pin = ''
        self.loadCap = 0
        self.loadRes = 0
        self.effectCap = [] # Rise Fall
        self.inslew = [] # Rise Fall
        self.outslew = [] # Rise Fall
        self.Delay = [] # Rise Fall
    def __repr__(self):
        effectCap_repr = ', '.join([f'{cap:.8f}' for cap in self.effectCap])
        inslew_repr = ', '.join([f'{slew:.8f}' for slew in self.inslew])
        outslew_repr = ', '.join([f'{slew:.8f}' for slew in self.outslew])
        delay_repr = ', '.join([f'{delay:.8f}' for delay in self.Delay])
        return f"CellArc(from_pin='{self.from_pin}', to_pin='{self.to_pin}', loadCap={self.loadCap:.8f}, loadRes={self.loadRes:.8f}, \neffectCap={{ {effectCap_repr} }}, \ninslew={{ {inslew_repr} }}, \noutslew={{ {outslew_repr} }}, \nDelay={{ {delay_repr} }})"

def zeroReplacer(a, b):
    if a == 'n/a':
        a = 0
    if b == 'n/a':
        b = 0
    return a, b
   
def Read_PtDelayRpt(inrpt):
    cellarcs = {}
    with open(inrpt, 'r') as infile:
        for line in infile:
            index = line.split()
            if line.startswith('From pin'):
                try:
                    newcellarc
                except NameError:
                    pass
                else:
                    if((newcellarc.from_pin, newcellarc.to_pin) not in cellarcs.keys()):
                        cellarcs[(newcellarc.from_pin, newcellarc.to_pin)] = newcellarc
                newcellarc = CellArc()
                newcellarc.from_pin = index[2]
            if line.startswith('To pin'):
                newcellarc.to_pin = index[2]
            if(len(index) > 4):
                if index[0] == 'Total' and index[1] == 'capacitance' and index[4] == '(in':
                    newcellarc.loadCap = float(index[3])
                elif index[0] == 'Total' and index[1] == 'resistance':
                    newcellarc.loadRes = float(index[3])
            if(len(index) > 6):
                if index[0] == 'Effective' and index[1] == 'capacitance' and index[6] == 'library':
                    index[3], index[4] = zeroReplacer(index[3], index[4])
                    newcellarc.effectCap = [float(index[3]), float(index[4])]
                elif index[0] == 'Input' and index[1] == 'transition' and index[2] == 'time':
                    index[4], index[5] = zeroReplacer(index[4], index[5])
                    newcellarc.inslew = [float(index[4]), float(index[5])]
                elif index[0] == 'Output' and index[1] == 'transition' and index[2] == 'time':
                    index[4], index[5] = zeroReplacer(index[4], index[5])
                    newcellarc.outslew = [float(index[4]), float(index[5])]
                elif index[0] == 'Cell' and index[1] == 'delay':
                    index[3], index[4] = zeroReplacer(index[3], index[4])
                    newcellarc.Delay = [float(index[3]), float(index[4])]
            if line.startswith('Rise delay'):
                newcellarc.Delay.append(float(index[3]))
            if line.startswith('Fall delay'):
                newcellarc.Delay.append(float(index[3]))
            if line.startswith('Rise transition'):
                newcellarc.outslew.append(float(index[3]))
            if line.startswith('Fall transition'):
                newcellarc.outslew.append(float(index[3]))
            if(len(index) > 3):
                if index[0] == '(X)' and index[1] == 'input_pin_transition':
                    if(len(newcellarc.inslew) < 3):
                        newcellarc.inslew.append(float(index[3]))
                if index[0] == '(Y)' and index[1] == 'output_net_total_cap':
                    newcellarc.loadCap = float(index[3])
                    newcellarc.effectCap = [0, 0]
        if((newcellarc.from_pin, newcellarc.to_pin) not in cellarcs.keys()):
            cellarcs[(newcellarc.from_pin, newcellarc.to_pin)] = newcellarc
    return cellarcs
class NetArc:
    def __init__(self):
        self.from_pin = ''
        self.to_pin = ''
        self.Delay = [] # Rise Fall
    def __repr__(self):
        delay_repr = ', '.join([f'{delay}' for delay in self.Delay])
        return f"NetArc(from_pin='{self.from_pin}', to_pin='{self.to_pin}', \nDelay={{ {delay_repr} }})"
    
def Read_SDF(insdf):
    netarcs = {}
    with open(insdf, 'r') as infile:
        for line in infile:
            index = line.split()
            if index[0] == '(INTERCONNECT':
                newnetarc = NetArc()
                newnetarc.from_pin = index[1]
                newnetarc.to_pin = index[2]
                if(len(index) < 5):
                    newnetarc.Delay = [float(index[3].split(':')[0].replace('(', ''))/0.95, float(index[3].split(':')[0].replace('(', ''))/0.95]
                if(len(index) == 5):
                    newnetarc.Delay = [float(index[3].split(':')[0].replace('(', ''))/0.95, float(index[4].split(':')[0].replace('(', ''))/0.95]
                netarcs[(newnetarc.from_pin, newnetarc.to_pin)] = newnetarc 
    return netarcs                    
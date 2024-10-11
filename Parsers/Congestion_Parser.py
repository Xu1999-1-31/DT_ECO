import re
import numpy as np

def ReadRouteCongestion(inrpt, scale):
    H_congestion = np.zeros((scale, scale))
    V_congestion = np.zeros((scale, scale))
    Layer_congestion = {}
    with open(inrpt, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            if line.startswith('##'):
                row = int(re.findall(r'\d+', line.split()[5])[0])
                column = int(re.findall(r'\d+', line.split()[6])[0])
            if re.match(r'^M\d+', line):
                if(line.split()[0] not in Layer_congestion):
                    Layer_congestion[line.split()[0]] = np.zeros((scale, scale))
                else:
                    Layer_congestion[line.split()[0]][row][column] = int(line.split()[2])
            if line.startswith('H routing'):
                H_congestion[row][column] = int(line.split()[3])
            if line.startswith('V routing'):
                V_congestion[row][column] = int(line.split()[3])
    
    return H_congestion, V_congestion, Layer_congestion
import re
import numpy as np

def ReadCellDensity(inrpt, scale):
    Cell_density = np.zeros((scale, scale))
    with open(inrpt, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            if line.startswith('##'):
                row = int(re.findall(r'\d+', line.split()[5])[0])
                column = int(re.findall(r'\d+', line.split()[6])[0])
            if line.startswith('Utilization Ratio'):
                Cell_density[row][column] = float(line.split()[2])
    
    return Cell_density
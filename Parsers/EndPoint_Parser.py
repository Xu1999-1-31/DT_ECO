def Read_EndPoint(inrpt, type):
    flag1 = False
    flag2 = False
    endPoint = {}
    with open(inrpt, 'r') as infile:
        for line in infile:
            index = line.split()
            if flag1 and flag2:
                if(len(index) > 0):
                    if '----' not in index[0] and index[0] != '1':
                        if type == 'untested':
                            endPoint[index[0]] = 0
                        elif type == 'met' or type == 'violated':
                            if index[0] not in endPoint.keys():
                                for i in range(len(index)):
                                    if index[i] == 'setup':
                                        if 'sdf_cond' not in index[i+1]:
                                            endPoint[index[0]] = float(index[i+1])
                                        else :
                                            endPoint[index[0]] = float(index[i+2])
            if(len(index) >= 3):
                if index[0] == 'Constrained' and index[1] == 'Related' and index[2] == 'Check':
                    flag1 = True
            if(len(index) >= 4):
                if index[0] == 'Pin' and index[1] == 'Pin' and index[2] == 'Clock' and index[3] == 'Type':
                    flag2 = True
    return endPoint
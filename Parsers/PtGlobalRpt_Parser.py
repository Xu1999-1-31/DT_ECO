import linecache

def Read_GlobalRpt(inrpt):
    linecache.clearcache()
    with open(inrpt, 'r') as infile:
        linecount = 0
        for line in infile:
            linecount += 1
            if line.startswith('Setup violations'):
                index = linecache.getline(inrpt, linecount+4).split()
                wns = float(index[1])
                index = linecache.getline(inrpt, linecount+5).split()
                tns = float(index[1])

    return wns, tns
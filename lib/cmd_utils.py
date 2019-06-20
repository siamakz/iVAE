import os
import sys

def seedify(arg_file):
    srange = [1, 1]
    args = sys.argv
    if len(args) == 1:
        pass
    elif len(args) == 2:
        srange = [1, int(args[1])]
    elif len(args) == 3:
        srange = [int(args[1]), int(args[2])]
    else:
        raise Exception('wrong usage')
    with open(arg_file, 'r') as f:
        seeded_name = arg_file.split('.txt')[0]+'_seeded.txt'
        with open(seeded_name, 'w') as sf:
            for line in f:
                for s in range(srange[0], srange[1]+1):
                    sf.write(line.split('\n')[0] + ' -s' + str(s) +'\n')


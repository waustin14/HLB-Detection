import os
import sys

LABELMAP = {
    '0': '5',
    '1': '6',
    '2': '5',
    '3': '6',
    '4': '4',
    '5': '5',
    '6': '1',
    '7': '0',
    '8': '2',
    '9': '3',
    '10': '7',
    '11': '5',
    '12': '4',
    '13': '4',
    '14': '4',
    '15': '7',
    '16': '7',
    '17': '6',
    '18': '7',
    '19': '4'
}

labelDir = './labels'

def relabel(labelDir, labelMap=LABELMAP):
    for labelFile in os.listdir(labelDir):
        try:
            with open(os.path.join(labelDir, labelFile), 'r') as f:
                lines = f.readlines()
            with open(os.path.join(labelDir, labelFile), 'w') as f:
                for line in lines:
                    line = line.strip()
                    fields = line.split(' ')
                    if fields[0] in labelMap.keys():
                        fields[0] = labelMap[fields[0]]
                    newLine = ' '.join(fields)
                    f.write(newLine + '\n')
        except Exception as e:
            print(f'Error while processing {labelFile}: {e}')
            continue

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        labelDir = args[1]
    relabel(labelDir)
    print('Relabeling done.')
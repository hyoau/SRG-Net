import numpy as np
import torch
use_cuda = True

data = []
xyz = []
normal = []
with open('data/demo_foot_clean.xyz', 'r') as file:
    for line in file.readlines():
        data.append([float(x) for x in line.split()])
        line = line.split()
        xyz.append([float(line[0])/5, float(line[1])/5, float(line[2])/5])
        normal.append([line[3], line[4], line[5]])
data = np.array(data).astype(np.float)
xyz = np.array(xyz).astype(np.float)

normal = np.array(normal).astype(np.float)

data_batch = [data, data]
xyz_batch = [xyz, xyz]


## hand_split
from util.add_label_to_part import get_hand_split_label

print('Start hand splitting...')
label = get_hand_split_label()
target = torch.from_numpy(np.array([label, label])).long()
print('target.shape', target.shape)
if use_cuda:
    target = target.cuda()

result = label
class_num = 6

## save result
label_colours = np.random.randint(255, size=(class_num, 3))
label_colours = [[237, 87, 54], [234, 255, 86], [130, 113, 0],
                    [72, 192, 163], [66, 76, 80], [68, 206, 246],
                    [209, 217, 224], [238, 222, 176], [98, 42, 29],
                    [65, 85, 93]]
label_colours = np.array(label_colours)
color_cloud = []

with open('result/hand_split_label.txt', 'w') as file:
    for i, item in enumerate(result):
        file.write(f'{item}\n')

for i in range(len(data)):
    line = [data[int(i)][0], data[int(i)][1], data[int(i)][2], *label_colours[int(result[i])]]
    color_cloud.append(line)

with open('result/hand_split_result.txt', 'w') as file:
    for i, line in enumerate(color_cloud):
        file.write(f'{line[0]}; {line[1]}; {line[2]}; {line[3]}; {line[4]}; {line[5]}\n')

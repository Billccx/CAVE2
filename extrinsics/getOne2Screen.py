import os

import numpy as np

t0s=[-420,-180,0]
t0s=np.array(t0s).reshape([3,1])


f=open('0toscreen.txt','r',encoding='utf-8')
R0s = []
knt = 0
for line in f:
    row = []
    line = line.strip()
    line = line.split(' ')
    row.append(float(line[0]))
    row.append(float(line[1]))
    row.append(float(line[2]))
    if (knt < 3):
        R0s.append(row)
    knt += 1
f.close()

R0s = np.array(R0s)

f2=open('1to0.txt','r',encoding='utf-8')
R10 = []
t10 = []
knt = 0
for line in f2:
    row = []
    line = line.strip()
    line = line.split(' ')
    row.append(float(line[0]))
    row.append(float(line[1]))
    row.append(float(line[2]))
    if (knt < 3):
        R10.append(row)
    else:
        t10.append(row)
    knt += 1
f2.close()

R10 = np.array(R10)
t10=np.array(t10).reshape([3,1])

R1s=np.matmul(R0s,R10)
t1s=np.matmul(R0s,t10)+t0s

f3=open('1toscreen.txt','w',encoding='utf-8')
for i in range(0,R1s.shape[0]):
    for j in range(0,R1s.shape[1]):
        f3.write(str(R1s[i][j])+' ')
    f3.write('\n')

f3.write(str(t1s[0][0])+' '+str(t1s[1][0])+' '+str(t1s[2][0])+'\n')
f3.close()




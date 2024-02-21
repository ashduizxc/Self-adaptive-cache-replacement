import pandas as pd

a = 150000
b = 150000
#c = 100000

df1 = pd.read_csv('soplex.csv',nrows = a - 1,header = 0)
df2 = pd.read_csv('roms.csv', nrows = b - 1,header = 0)
#df3 = pd.read_csv('bzips1.csv', nrows = c - 1,header = 0)

df1.to_csv('test11'+'.csv',mode='a',header=1,index=0)
df2.to_csv('test11'+'.csv',mode='a',header=1,index=0)
#df3.to_csv('test22'+'.csv',mode='a',header=1,index=0)

#1 astar+gem
#2 libquantum+gromacs
#3 bwave+cam4
#4 bzips+lbm
#5 cactus+mlic
#6 omne+gems
#7 gromacs+h264ref
#8 soplex+roms
#9 gcc+spginx3
#10 xalancbmk+mcf1

#11 soplex+roms
#12 mcf1+cactusADM
#13 omnetpp+xalancbmk
#14 mcf1+lbm
#15 gcc+spginx3
#16 bwave+bzip
#17 cam4+xalancbmk
#18 soplex+gcc
#19 bzips+lbm
#20 gems+xalancbmk+gcc
#21 spginx3+bzips+mcf1
#22 astar+roms+bzips
#23 omnetpp+cam4+cactusADM
#24 cam4+astar

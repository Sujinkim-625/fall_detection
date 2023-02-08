import os



f=open("test.csv", 'w')
f.write("video,frame,label\n")

for i in range(1,245):
    if 12 < i and 80:
       f.write("test.csv,{0},{0}\n".format(i, 0))
    else:
       f.write("test.csv,{0},{0}\n".format(i, 2))

f.close()
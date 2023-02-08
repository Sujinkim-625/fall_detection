import cv2
import os
import time

spath='E:/frame_division/'
file_list=os.listdir(spath+'Original_senior/')

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs('C:/Users/Public.DESKTOP-FLI5G44/Desktop/openpose/_output/'+directory)
    except OSError:
        print ('Make : %s Folder'%directory)

# def video2frame(invideofilename, save_path):
#     vidcap = cv2.VideoCapture(invideofilename)
#     count = 0
#     while True:
#       success,image = vidcap.read()
#       if not success:
#           break
#       print ('Read a new frame: ', success)
#       sp=invideofilename.split('/')
#       fname = sp[-1][:-4]+"_"+"{}.png".format("{0:06d}".format(count))
#       cv2.imwrite(save_path + fname, image) # save frame as JPEG file
#       count += 1
#     print("{} images are extracted in {}.". format(count, save_path))

#print(file_list)
for i in file_list:
  createFolder('%s'%(i[0:-4]))
#   time.sleep(2)
#   video2frame(spath+'Original_senior/'+i,spath+i[0:-4]+'/')


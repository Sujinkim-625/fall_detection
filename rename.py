import os 

file_path = "C:\Users\CSDC\Desktop\Human-Falling-Detect-Tracks-master\Data\falldata\Coffee_room_01\Annotation_files"
file_name_list = os.listdir(file_path)

i=1
for name in file_name_list:
    src = os.path.join(file_path, name)
    extend = name.split(".")[1]
    dst = str(i) + '.' + extend
    os.rename(src, dst)
    i += 1
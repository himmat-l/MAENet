import os

dir_lists = ['image','depth','label37']
cur_dir = os.getcwd()
file_lists = []
# val_dir = os.path.join(cur_dir,'val')
# file_list = os.listdir(val_dir)
val_dir = []

for dir in dir_lists:
    tmp_dir = os.path.join(cur_dir,dir,'val')
    file_list = sorted(os.listdir(tmp_dir))
    val_dir.append(tmp_dir)
    file_lists.append(file_list)

with open('val37.txt','r') as f:
    for index in range(len(file_lists[0])):
        file_path0 = os.path.join(val_dir[0],file_lists[0][index])
        file_path1 = os.path.join(val_dir[1], file_lists[1][index])
        file_path2 = os.path.join(val_dir[2], file_lists[2][index])
        f.write(file_path0 + ' ' + file_path1 + ' ' + file_path2 + ' ' + '\n')

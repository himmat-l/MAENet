import os
import shutil
import numpy as np

def replace():
	"""
	修改文件内容，替换文件路径
	:return:
	"""
	f1 = '/home/liuxiaohui/MAENet/data/NYUDv2/test.txt'
	f2 = '/home/liuxiaohui/MAENet/data/NYUDv2/ntest.txt'
	# txt_file_selinux = file_selinux + '.txt'
	# temp_file_selinux = file_selinux + '.txt'

	if not os.path.exists(f1):
		exit(-1)
	# shutil.copy2(file_selinux, backup_file_selinux)
	lines = open(f1,'r').readlines()
	# print(lines)
	f = open(f2,'w')

	for s in lines:
		# print(s)
		s = s.replace('/home/liuxiaohui/sharon_l-ACNet-master/ACNet/data/nyuv2','/home/liuxiaohui/MAENet/data/NYUDv2')
		# print(s)
		f.write(s)
	f.close()
	return 0

if __name__ == '__main__':
	replace()
import os

for path, subdirs, files in os.walk("../Output/greedy/"):
    for f in files:
	file_name = os.path.join(path,  f)
	print(file_name) 
	os.rename(file_name, file_name.split(".")[0] + ".Temporal"+".xml")


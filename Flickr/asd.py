from os import walk
mypath = 'instance/test/'
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    
print(f[0])

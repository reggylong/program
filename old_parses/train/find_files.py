import os

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

os.chdir("iter.4")
for filename in os.listdir(os.getcwd()):
    if "oracle" in filename:
        continue
    length = file_len(filename)
    if length < 10:
        print filename

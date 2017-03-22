import bcolz
from glob import glob

DIR = "/data/dsb3/stage1+luna_bcolz/"

files = glob(DIR+"*")
print len(files)

for f in files:
    print f
    a = bcolz.open(f, mode="r")[:]
    print a.shape, a.dtype
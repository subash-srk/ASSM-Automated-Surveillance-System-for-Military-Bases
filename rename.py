import os
os.chdir('data/val/AFFECTED')
i=1
for file in os.listdir():
    src=file
    dst="Apple_Affected"+"_"+str(i)+".jpg"
    os.rename(src,dst)
    i+=1


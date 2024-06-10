
"""
freesurfer loop run

example command :
#export FREESURFER_HOME=/home/vor/freesurfer
#source $FREESURFER_HOME/SetUpFreeSurfer.sh

#python3 ./freesurfer.py /media/vor/NIIDRIVE

output :
/home/vor/freesurfer/subjects

"""

import subprocess, os, sys
from os.path import join

storePath = (sys.argv[1])
# storePath = r"/home/vor/projects/data/MRI/out/I"

print('start freesufe' ,storePath)
for root, subdirs, files in os.walk(storePath):
    for f in files:
        if f.endswith('.nii'):
            fileName = os.path.splitext(f)[0]
            fileInput = join(root, f)
            fileOut = join (root,fileName)
            print(fileInput)
            command = ('recon-all' ,'-s' ,fileOut ,'-i', fileInput,'-autorecon1','-autorecon2', '-openmp 8')
            cmd = ' '.join(command)

            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in p.stdout.readlines():
                print(line.strip())
            retval = p.wait()

print('End Freesurfer Loop', storePath)


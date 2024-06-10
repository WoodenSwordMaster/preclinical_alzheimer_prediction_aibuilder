import os, shutil, sys

def find_best_nii(standardpath,finalpath):
    for sub_root, sub_dirs, sub_files in os.walk(standardpath):
        pathList = sub_root.split(os.sep)
        for g in pathList:
            a = g.find("_I")
            k = 0
            if a != -1:
                k = 1
            if k == 1 or g.startswith("I"):
                if sub_root.endswith(g):
                    #print(sub_root)
                    new_path = sub_root.replace(standardpath,finalpath)
                    print(sub_root)
                    for sub_root2, sub_dirs2, sub_files2 in os.walk(sub_root):
                        for file in sub_files2:
                            if file == 'brain.mgz':
                                og_path = os.path.join(sub_root2,file)
                                print(og_path, "...to...", new_path)
                                if not os.path.exists(new_path):
                                    os.makedirs(new_path)
                                shutil.copy2(og_path, new_path)




# storePath = (sys.argv[1])
# endPath = (sys.argv[2])
# find_best_nii(standardpath=storePath,finalpath=endPath)

find_best_nii(standardpath=r"D:\ADNI_Dataset_output\MCI_AD_2",finalpath=r"D:\brain_mgz_only1\MCI_AD_2")




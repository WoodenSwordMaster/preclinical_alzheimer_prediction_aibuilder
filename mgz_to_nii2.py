import os
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting

storePath = r"D:\brain_mgz_only1\CN_2"
endPath = r"D:\final_dataset2\CN"


for root, subdirs, files in os.walk(storePath):
    for f in files:
        if f.endswith('.mgz'):
            start_path = os.path.join(root, f)
            new_path = start_path.replace(storePath, endPath)
            pathList = root.split(os.sep)
            Ipath = "brain"
            for g in pathList:
                a = g.find("_I")
                k = 0
                if a != -1:
                    k = 1
                if k == 1 or g.startswith("I"):
                    Ipath = g
            new_path = new_path.replace(f, Ipath + ".nii")
            print(start_path, ".....start_path")
            print(new_path, ".....end_path")
            if not os.path.exists(new_path.replace(Ipath + ".nii", "")):
                os.makedirs(new_path.replace(Ipath + ".nii", ""))
            img = nib.load(start_path)

            # fig, ax = plt.subplots(figsize=[10, 5])
            # plotting.plot_img(img, cmap='gray', axes=ax, display_mode='mosaic')  # , cut_coords=(0, 0, 0))
            # plt.show()

            nib.save(img, new_path)
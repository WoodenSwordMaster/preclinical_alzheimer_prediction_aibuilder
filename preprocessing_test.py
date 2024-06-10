import os
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting

data_path = r"C:\Users\punnut\Downloads\final_dataset+AD_1+2\CN\I10897"
#out_path = r"C:\Users\punnut\Downloads\test2/test.nii.gz"
for root, dir,files in os.walk(data_path):
    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            nii_path = root + "/" + file
            nii_img = nib.load(nii_path)
            # nii_img2 = nii_img.get_fdata()
            # print(nii_img2)
            # print(nii_img2.shape)
            print(file, "==", nii_img.shape)
            fig, ax = plt.subplots(figsize=[10, 5])
            plotting.plot_img(nii_img, cmap='gray', axes=ax, display_mode='mosaic')#, cut_coords=(0, 0, 0))
            plt.show()

            #nib.save(nii_img, out_path)

import os
import nibabel as nib
from scipy import ndimage
import numpy as np
import torchio as tio

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize2(volume):
    # Normalize the volume to zero mean and unit variance
    volume = (volume - np.mean(volume)) / np.std(volume)  # <-- Added normalization
    volume = volume.astype("float32")
    return volume

def resize_volume(img, resize):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = resize
    desired_width = resize
    desired_height = resize
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 180, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path, resize):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize2(volume)
    # Resize width, height and depth
    volume = resize_volume(volume, resize)
    return volume


data_path = r"C:\Users\punnut\Downloads\final_dataset+AD_2"
final_path = r"C:\Users\punnut\Downloads\final_dataset1+2_preprocessed3+AD_traintest_96"
AD_class = []
CN_class = []
MCI_class = []
MCI_AD_class = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".nii"):
            pathList = root.split(os.sep)
            for i in pathList:
                if i == "AD":
                    AD_class.append(os.path.join(root,file))
                    #print(os.path.join(root,file))
                elif i == "CN":
                    CN_class.append(os.path.join(root, file))
                    #print(os.path.join(root, file))
                elif i == "MCI":
                    MCI_class.append(os.path.join(root, file))
                    #print(os.path.join(root, file))
                elif i == "MCI_AD":
                    MCI_AD_class.append(os.path.join(root, file))
                    #print(os.path.join(root, file))

from sklearn.model_selection import train_test_split
all_list = []
if len(AD_class) != 0:
    all_list.append(AD_class)
if len(MCI_class) != 0:
    all_list.append(MCI_class)
if len(MCI_AD_class) != 0:
    all_list.append(MCI_AD_class)
if len(CN_class) != 0:
    all_list.append(CN_class)

resize = 96

for list in all_list:
    split_list = []
    for path in list:
        file = process_scan(path, resize)
        print(path,"processed")
        array = [file,path]
        split_list.append(array)
    train, val = train_test_split(split_list, test_size=0.2, random_state=42)
    print("finished split train val")

    for array in train:
        print(array[1])
        new_path = array[1].replace(data_path,final_path + "/train")
        if new_path.find("CN_I") != -1:
            string_list = new_path.partition("CN_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]
        elif new_path.find("MCI_I") != -1:
            string_list = new_path.partition("MCI_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]
        elif new_path.find("MCI_AD_I") != -1:
            string_list = new_path.partition("MCI_AD_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]
        elif new_path.find("AD_I") != -1:
            string_list = new_path.partition("AD_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]

        data = array[0]
        data = np.expand_dims(data, axis=0)
        preprocess_list2 = [tio.RandomGamma(log_gamma=(-0.3, 0.3)),
                        tio.RandomSwap(patch_size=int(resize/8),num_iterations=15)]
        transform = tio.Compose(preprocess_list2)
        print(transform)
        print("os.path", new_path)

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        print("train transforming", os.path.join(new_path, file))
        n = 6
        for i in range(1, n + 1):
            data_n = transform(data)
            data_n = np.reshape(data_n,(resize,resize,resize))
            data_n = np.expand_dims(data_n, axis=-1)
            image_n = nib.Nifti1Image(data_n, affine=np.eye(4))
            nib.save(image_n, os.path.join(new_path, file.replace(".nii", "_" + str(i) +".nii")))
            print("train transformed", i)

    for array in val:
        new_path = array[1].replace(data_path, final_path + "/test")
        if new_path.find("CN_I") != -1:
            string_list = new_path.partition("CN_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]
        elif new_path.find("MCI_I") != -1:
            string_list = new_path.partition("MCI_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]
        elif new_path.find("MCI_AD_I") != -1:
            string_list = new_path.partition("MCI_AD_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]
        elif new_path.find("AD_I") != -1:
            string_list = new_path.partition("AD_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        data = array[0]
        data = np.expand_dims(data, axis=-1)
        image_n = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(image_n, os.path.join(new_path, file))
        print("vaild transformed", os.path.join(new_path, file))





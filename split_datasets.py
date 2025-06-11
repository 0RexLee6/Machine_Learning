import os, shutil, random

src_cat = r'C:\Users\s8104\Desktop\Univerity_of_Taipei\3rd_second_semester\ML\PetImages\Cat'
src_dog = r'C:\Users\s8104\Desktop\Univerity_of_Taipei\3rd_second_semester\ML\PetImages\Dog'
dst_root = r'C:\Users\s8104\Desktop\Univerity_of_Taipei\3rd_second_semester\ML\data_split'
os.makedirs(os.path.join(dst_root, 'train', 'cat'), exist_ok=True)
os.makedirs(os.path.join(dst_root, 'train', 'dog'), exist_ok=True)
os.makedirs(os.path.join(dst_root, 'val', 'cat'), exist_ok=True)
os.makedirs(os.path.join(dst_root, 'val', 'dog'), exist_ok=True)

def split_and_copy(src, dst_train, dst_val, ratio=0.8):
    files = [f for f in os.listdir(src) if f.endswith('.jpg')]
    random.shuffle(files)
    split = int(len(files) * ratio)
    for i, fname in enumerate(files):
        try:
            srcfile = os.path.join(src, fname)
            if i < split:
                shutil.copy(srcfile, os.path.join(dst_train, fname))
            else:
                shutil.copy(srcfile, os.path.join(dst_val, fname))
        except:
            # 某些圖片壞掉可略過
            continue

split_and_copy(src_cat, os.path.join(dst_root, 'train', 'cat'), os.path.join(dst_root, 'val', 'cat'))
split_and_copy(src_dog, os.path.join(dst_root, 'train', 'dog'), os.path.join(dst_root, 'val', 'dog'))

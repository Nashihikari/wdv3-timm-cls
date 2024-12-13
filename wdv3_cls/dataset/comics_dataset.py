import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import albumentations
import glob
import time
import pdb
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import zoom, convolve, binary_dilation, grey_dilation 
import copy
from tqdm import tqdm

KERNEL_SIZE = np.array([
    [1, 1],
    [1, 1],
])

def resize_array(image, target=(512, 512)):

    # if len(image.shape) == 2:
    #     oh, ow = image.shape
    #     zoom_factors = (oh // target[0], ow // target[1])
    #     image = zoom(image, zoom_factors)
    # elif len(image.shape) == 3:
    #     oh, ow, _ = image.shape
    #     zoom_factors = (oh // target[0], ow // target[1], 1)
    #     image = zoom(image, zoom_factors)
    image = Image.fromarray(image)
    image = image.resize(target, Image.BILINEAR)

    return np.array(image)

def get_parquet_2dcomics(data_dir, save_dir ,type_data='train'):
    time_start = time.time()

    assert type_data in ['train', 'val', 'test']

    parquet_save_dir = os.path.join(save_dir, f'2dcomics_{type_data}.parquet')
    img_data_list = os.listdir(data_dir)
    img_data_list = [img_data.split('.')[0] for img_data in img_data_list]
    if type_data == 'train':
        img_data_list = [img_data for img_data in img_data_list if int(img_data.split('_')[-1]) % 10 != 0]
    else:
        img_data_list = [img_data for img_data in img_data_list if int(img_data.split('_')[-1]) % 10== 0]

    df = pd.DataFrame({'filename': img_data_list})
    df.info()
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_save_dir)
    time_end = time.time()
    time_cost = time_end - time_start
    return time_cost

def check_parquet_2dcomics(data_dir):
    time_start = time.time()

    data_parquet = pd.read_parquet(data_dir)
    data_info = data_parquet.info()

    time_end = time.time()
    time_cost = time_end - time_start

    return data_info, time_cost

class ComicsLcLDataset(Dataset):
    def __init__(self, data_dir, img_filename_parquet, transform=None, line_concat=None, lc_avg=None):
        self.data_dir = data_dir
        self.filename_df = pd.read_parquet(img_filename_parquet)
        self.transform = transform
        self.l_dir = 'l'
        self.lc_dir = 'lc'
        self.lc_avg_dir = 'lc_avg' if lc_avg == None else lc_avg
        self.pc_ID_dir = 'person_clothes_ID_map'
        self.line_concat = line_concat
        self.cp = 'cp'
        # tmp
        self.image_color = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/Images_color'
        self.pc_ID_tmp = '/data/liuhaoxiang/data_97/comics_local/person_clothes_ID_map'
        print("SuccessFul Init ComicsLcAvgLDataset")

    def __len__(self):
        return len(self.filename_df)

    def __getitem__(self, index):
        '''
            Every item in the dataset, select one of the person_clothes_id
        '''
        # tmp
        img_path = os.path.join(self.data_dir ,  self.lc_dir ,self.filename_df.iloc[index]['filename'] + '.png')
        img_l_path = os.path.join(self.data_dir, self.l_dir ,self.filename_df.iloc[index]['filename'] + '.png')

        image_l = Image.open(img_l_path).convert("RGB")

        image = np.array(Image.open(img_path).convert("RGB"))
        cp_mask = np.load(os.path.join(self.data_dir, self.cp, self.filename_df.iloc[index]['filename']+'.npy'))
        pc_ID_map = np.load(os.path.join(self.data_dir, self.pc_ID_dir, self.filename_df.iloc[index]['filename']+'.npy'))


        image = Image.fromarray(image).convert("RGB")
        image = image.resize((image_l.size))
        image = np.array(image)
        image_l = np.array(image_l)
        # 为了节省时间。
        # image = resize_array(image)
        # pc_ID_map = resize_array(pc_ID_map)
        # cp_mask = resize_array(cp_mask)

        pc_unique_ID = np.unique(pc_ID_map)
        pc_unique_ID = pc_unique_ID[pc_unique_ID != 0]
        pc_ID = np.random.choice(pc_unique_ID)
        # print(pc_ID)

        cp_mask_unique = np.unique(cp_mask)
        cp_mask_max = np.max(cp_mask_unique)
        cp_mask[cp_mask != cp_mask_max] = 0
        cp_mask[cp_mask == cp_mask_max] = 1

        pc_ID_map[pc_ID_map != pc_ID] = 0
        pc_ID_map[pc_ID_map == pc_ID] = 1

        pc_ID_map = pc_ID_map - cp_mask

        mask_ID_map = np.stack([pc_ID_map, pc_ID_map, pc_ID_map], axis=2)
        # print(image.shape, mask_ID_map.shape, cp_mask.shape)

        image = np.where(mask_ID_map == 1, image, 255)
        image_l = np.where(mask_ID_map == 1, image_l, 255)

        rows, cols = np.where(pc_ID_map == 1)
        # 计算bbox的左上角和右下角
        top_left = (np.min(rows), np.min(cols))
        bottom_right = (np.max(rows), np.max(cols))
        # top_left = (np.min(cols), np.min(rows))
        # bottom_right = (np.max(cols), np.max(rows))
        image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]
        image_l = image_l[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]


        # image_1 = Image.fromarray(image).convert('RGB')
        # image_1.save(f"test_img_3/{self.filename_df.iloc[index]['filename']}.png")
        # pdb.set_trace()
        if self.transform is not None:
            augmentations = self.transform(image=image)
            augmentations_l = self.transform(image=image_l)

            image = augmentations['image']
            image_l = augmentations_l['image']
            image = image[[2, 1, 0]]
            image_l = image_l[[2, 1, 0]]

            image = torch.cat((image_l, image), dim=0)

        return image, int(pc_ID)


class ComicsLcAvgLDataset(Dataset):
    def __init__(self, data_dir, img_filename_parquet, transform=None, line_concat=None, lc_avg=None):
        self.data_dir = data_dir
        self.filename_df = pd.read_parquet(img_filename_parquet)
        self.transform = transform
        self.l_dir = 'l'
        self.lc_dir = 'lc'
        self.lc_avg_dir = 'lc_avg' if lc_avg == None else lc_avg
        self.pc_ID_dir = 'person_clothes_ID_map'
        self.line_concat = line_concat
        self.cp = 'cp'
        # tmp
        self.image_color = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/Images_color'
        self.pc_ID_tmp = '/data/liuhaoxiang/data_97/comics_local/person_clothes_ID_map'
        print("SuccessFul Init ComicsLcAvgLDataset")

    def __len__(self):
        return len(self.filename_df)

    def __getitem__(self, index):
        '''
            Every item in the dataset, select one of the person_clothes_id
        '''
        # tmp
        img_path = os.path.join(self.image_color ,self.filename_df.iloc[index]['filename'] + '.png')
        img_l_path = os.path.join(self.data_dir, self.l_dir ,self.filename_df.iloc[index]['filename'] + '.png')

        image_l = Image.open(img_l_path).convert("RGB")

        image = np.array(Image.open(img_path).convert("RGB"))
        cp_mask = np.load(os.path.join(self.data_dir, self.cp, self.filename_df.iloc[index]['filename']+'.npy'))
        pc_ID_map = np.load(os.path.join(self.data_dir, self.pc_ID_dir, self.filename_df.iloc[index]['filename']+'.npy'))


        image = Image.fromarray(image).convert("RGB")
        image = image.resize((image_l.size))
        image = np.array(image)
        image_l = np.array(image_l)
        # 为了节省时间。
        # image = resize_array(image)
        # pc_ID_map = resize_array(pc_ID_map)
        # cp_mask = resize_array(cp_mask)

        pc_unique_ID = np.unique(pc_ID_map)
        pc_unique_ID = pc_unique_ID[pc_unique_ID != 0]
        pc_ID = np.random.choice(pc_unique_ID)
        # print(pc_ID)

        cp_mask_unique = np.unique(cp_mask)
        cp_mask_max = np.max(cp_mask_unique)
        cp_mask[cp_mask != cp_mask_max] = 0
        cp_mask[cp_mask == cp_mask_max] = 1

        pc_ID_map[pc_ID_map != pc_ID] = 0
        pc_ID_map[pc_ID_map == pc_ID] = 1

        pc_ID_map = pc_ID_map - cp_mask

        mask_ID_map = np.stack([pc_ID_map, pc_ID_map, pc_ID_map], axis=2)
        # print(image.shape, mask_ID_map.shape, cp_mask.shape)

        image = np.where(mask_ID_map == 1, image, 255)
        image_l = np.where(mask_ID_map == 1, image_l, 255)

        rows, cols = np.where(pc_ID_map == 1)
        # 计算bbox的左上角和右下角
        top_left = (np.min(rows), np.min(cols))
        bottom_right = (np.max(rows), np.max(cols))
        # top_left = (np.min(cols), np.min(rows))
        # bottom_right = (np.max(cols), np.max(rows))
        image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]
        image_l = image_l[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]


        # image_1 = Image.fromarray(image).convert('RGB')
        # image_1.save(f"test_img_3/{self.filename_df.iloc[index]['filename']}.png")
        # pdb.set_trace()
        if self.transform is not None:
            augmentations = self.transform(image=image)
            augmentations_l = self.transform(image=image_l)

            image = augmentations['image']
            image_l = augmentations_l['image']
            image = image[[2, 1, 0]]
            image_l = image_l[[2, 1, 0]]

            image = torch.cat((image_l, image), dim=0)

        return image, int(pc_ID)


class ComicsDataset(Dataset):
    def __init__(self, data_dir, img_filename_parquet, transform=None, line_concat=None, lc_avg='Images_color'):
        self.data_dir = data_dir
        self.filename_df = pd.read_parquet(img_filename_parquet)
        self.transform = transform
        self.l_dir = 'l'
        self.lc_dir = 'lc'
        self.lc_avg_dir = 'lc_avg' if lc_avg == None else lc_avg
        self.pc_ID_dir = 'person_clothes_ID_map'
        self.line_concat = line_concat
        self.cp = 'cp'
        # tmp
        self.image_color = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/Images_color'
        self.pc_ID_tmp = '/data/liuhaoxiang/data_97/comics_local/person_clothes_ID_map'


    def __len__(self):
        return len(self.filename_df)

    def __getitem__(self, index):
        '''
            Every item in the dataset, select one of the person_clothes_id
        '''
        # img_lc_path = os.path.join(self.data_dir, self.lc_avg_dir, self.filename_df.iloc[index]['filename'] + '.png')
        img_l_path = os.path.join(self.data_dir, self.l_dir ,self.filename_df.iloc[index]['filename'] + '.png')

        image_l = Image.open(img_l_path).convert("RGB")

        image = np.array(image_l)
        cp_mask = np.load(os.path.join(self.data_dir, self.cp, self.filename_df.iloc[index]['filename']+'.npy'))
        pc_ID_map = np.load(os.path.join(self.data_dir, self.pc_ID_dir, self.filename_df.iloc[index]['filename']+'.npy'))

        pc_unique_ID = np.unique(pc_ID_map)
        pc_unique_ID = pc_unique_ID[pc_unique_ID != 0]
        pc_ID = np.random.choice(pc_unique_ID)
        # print(pc_ID)

        cp_mask_unique = np.unique(cp_mask)
        cp_mask_max = np.max(cp_mask_unique)
        cp_mask[cp_mask != cp_mask_max] = 0
        cp_mask[cp_mask == cp_mask_max] = 1

        pc_ID_map[pc_ID_map != pc_ID] = 0
        pc_ID_map[pc_ID_map == pc_ID] = 1

        pc_ID_map = pc_ID_map - cp_mask

        mask_ID_map = np.stack([pc_ID_map, pc_ID_map, pc_ID_map], axis=2)
        # print(image.shape, mask_ID_map.shape, cp_mask.shape)

        image = np.where(mask_ID_map == 1, image, 255)
        rows, cols = np.where(pc_ID_map == 1)
        # 计算bbox的左上角和右下角
        top_left = (np.min(rows), np.min(cols))
        bottom_right = (np.max(rows), np.max(cols))
        # top_left = (np.min(cols), np.min(rows))
        # bottom_right = (np.max(cols), np.max(rows))
        image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]
        
        image_l = Image.fromarray(image)

        image_l_dilated = image_l.convert('L')
        image_l_dilated = np.array(image_l_dilated)
        # image_l_dilated = convolve(image_l_dilated, KERNEL_SIZE, mode='constant', cval=0)  
        image_l_dilated = 255. - image_l_dilated
        image_l_dilated = grey_dilation(image_l_dilated, size=(3,3))
        image_l_dilated = grey_dilation(image_l_dilated, size=(3,3))

        image_l_dilated = 255. - image_l_dilated
        image_l_dilated = Image.fromarray(image_l_dilated).convert('RGB')

        # image_1 = Image.fromarray(image).convert('RGB')
        # image_1.save(f"test_img_3/{self.filename_df.iloc[index]['filename']}.png")
        # pdb.set_trace()
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations['image']
            image = image[[2, 1, 0]] # RGB=>BGR


        return image, int(pc_ID)

    def data_clear(self, index):
        '''
            Every item in the dataset, select one of the person_clothes_id
        '''
        try:

            # tmp
            img_lc_path = os.path.join(self.data_dir, self.lc_avg_dir, self.filename_df.iloc[index]['filename'] + '.png')
            img_l_path = os.path.join(self.data_dir, self.l_dir ,self.filename_df.iloc[index]['filename'] + '.png')

            image_l = Image.open(img_l_path).convert("RGB")

            image = np.array(image_l)
            cp_mask = np.load(os.path.join(self.data_dir, self.cp, self.filename_df.iloc[index]['filename']+'.npy'))
            pc_ID_map = np.load(os.path.join(self.data_dir, self.pc_ID_dir, self.filename_df.iloc[index]['filename']+'.npy'))

            pc_unique_ID = np.unique(pc_ID_map)
            pc_unique_ID = pc_unique_ID[pc_unique_ID != 0]
            pc_ID = np.random.choice(pc_unique_ID)
            # print(pc_ID)

            cp_mask_unique = np.unique(cp_mask)
            cp_mask_max = np.max(cp_mask_unique)
            cp_mask[cp_mask != cp_mask_max] = 0
            cp_mask[cp_mask == cp_mask_max] = 1

            pc_ID_map[pc_ID_map != pc_ID] = 0
            pc_ID_map[pc_ID_map == pc_ID] = 1

            pc_ID_map = pc_ID_map - cp_mask

            mask_ID_map = np.stack([pc_ID_map, pc_ID_map, pc_ID_map], axis=2)
            # print(image.shape, mask_ID_map.shape, cp_mask.shape)

            image = np.where(mask_ID_map == 1, image, 255)
            rows, cols = np.where(pc_ID_map == 1)
            # 计算bbox的左上角和右下角
            top_left = (np.min(rows), np.min(cols))
            bottom_right = (np.max(rows), np.max(cols))
            # top_left = (np.min(cols), np.min(rows))
            # bottom_right = (np.max(cols), np.max(rows))
            image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]

            # image_1 = Image.fromarray(image).convert('RGB')
            # image_1.save(f"test_img_3/{self.filename_df.iloc[index]['filename']}.png")
            # pdb.set_trace()
            if self.transform is not None:
                augmentations = self.transform(image=image)
                image = augmentations['image']
                image = image[[2, 1, 0]] # RGB=>BGR
        except:
            print(f"Error: {self.filename_df.iloc[index]['filename']}")
            self.filename_df = self.filename_df.drop(index)

        return image, int(pc_ID)

    def image_after_mask(self, index):
        '''
            Every item in the dataset, select one of the person_clothes_id
        '''


        img_path = os.path.join(self.data_dir, self.lc_dir ,self.filename_df.iloc[index]['filename'] + '.png')
        img_path_avg = os.path.join(self.image_color ,self.filename_df.iloc[index]['filename'] + '.png')
        image_path_l = os.path.join(self.data_dir, self.l_dir ,self.filename_df.iloc[index]['filename'] + '.png')
        
        pc_ID_map = np.load(os.path.join(self.data_dir, self.pc_ID_dir, self.filename_df.iloc[index]['filename']+'.npy'))

        image = Image.open(img_path).convert("RGB")
        image.save(f"test_img/origin_{self.filename_df.iloc[index]['filename']}.png")

        image_lc_avg = Image.open(img_path_avg).convert("RGB")
        image_lc_avg = image_lc_avg.resize((image.size))

        image = np.array(image)
        image_lc_avg = np.array(image_lc_avg)
        image_l = np.array(Image.open(image_path_l).convert("RGB"))
        print(self.filename_df.iloc[index]['filename'])
        print(image.shape, image_lc_avg.shape, image_l.shape)
        cp_mask = np.load(os.path.join(self.data_dir, self.cp, self.filename_df.iloc[index]['filename']+'.npy'))

        # pdb.set_trace()
        # randomly select one of the person_clothes_id
        # image = resize_array(image)
        # pc_ID_map = resize_array(pc_ID_map)

        pc_unique_ID = np.unique(pc_ID_map)
        pc_unique_ID = pc_unique_ID[pc_unique_ID != 0]
        pc_ID = np.random.choice(pc_unique_ID)

        cp_mask_unique = np.unique(cp_mask)
        cp_mask_max = np.max(cp_mask)
        cp_mask[cp_mask != cp_mask_max] = 0
        cp_mask[cp_mask == cp_mask_max] = 1

        pc_ID_map[pc_ID_map != pc_ID] = 0
        pc_ID_map[pc_ID_map != 0] = 1
        pc_ID_map_copy = copy.deepcopy(pc_ID_map)

        pc_ID_map = pc_ID_map - cp_mask

        rows, cols = np.where(pc_ID_map == 1)
        # 计算bbox的左上角和右下角
        top_left = (np.min(rows), np.min(cols))
        bottom_right = (np.max(rows), np.max(cols))
        # top_left = (np.min(cols), np.min(rows))
        # bottom_right = (np.max(cols), np.max(rows))
        print(top_left, bottom_right)
        mask_ID_map = np.stack([pc_ID_map, pc_ID_map, pc_ID_map], axis=2)

        image_mask = np.where(mask_ID_map == 1, image, 255)
        image_mask = Image.fromarray(image_mask).convert('RGB')
        image_mask.save(f"test_img_2/mask_{self.filename_df.iloc[index]['filename']}.png")

        # image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]
        # image = Image.fromarray(image).convert('RGB')
        # image.save(f"test_img/{self.filename_df.iloc[index]['filename']}{time.time()}.png")


        image_lc_avg = np.where(mask_ID_map == 1, image_lc_avg, 255)
        image_lc_avg = image_lc_avg[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]
        image_lc_avg = Image.fromarray(image_lc_avg).convert('RGB')
        image_lc_avg.save(f"test_img_2/lc_avg_{self.filename_df.iloc[index]['filename']}.png")

        image_l = np.where(mask_ID_map == 1, image_l, 255)
        image_l = image_l[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]





        image_l = Image.fromarray(image_l).convert('RGB')
        image_l.save(f"test_img_2/l_{self.filename_df.iloc[index]['filename']}.png")

        image_l_dilated = image_l.convert('L')
        image_l_dilated = np.array(image_l_dilated)
        # image_l_dilated = convolve(image_l_dilated, KERNEL_SIZE, mode='constant', cval=0)  
        image_l_dilated = 255. - image_l_dilated
        image_l_dilated = grey_dilation(image_l_dilated, size=(3,3))
        image_l_dilated = grey_dilation(image_l_dilated, size=(3,3))

        image_l_dilated = 255. - image_l_dilated
        image_l_dilated = Image.fromarray(image_l_dilated).convert('RGB')
        image_l_dilated.save(f"test_img_2/l_dilated_{self.filename_df.iloc[index]['filename']}.png")
        # time_3 = time.time()
        
        # if self.transform is not None:
        #     augmentations = self.transform(image=image)
        #     image = augmentations['image']
            
        #     image = image[[2, 1, 0]]
        # time_4 = time.time()
        # print(time_1 - time_0, time_2 - time_1, time_3 - time_2, time_4 - time_3)
        return image, int(pc_ID)


class MengbaoDataset(Dataset):
    def __init__(self, mengbao_dir, comics_dir,  other_work_parquet, transform=None, line_concat=None, lc_avg='Images_color', stage_type='train', other_work=True):
        '''
            tmp data dir: /data/liuhaoxiang/data_97/Mengbao_Dataset/training_set_20241129_aug
            tmp data dir: /data/liuhaoxiang/longhao_97/share/mengbao
        '''
        #########################################################
        # mengbao
        self.other_work = other_work
        if not self.other_work:
            print("Current Stage: Training Without Other work")
        self.mengbao_dir = mengbao_dir
        self.transform = transform
        self.l_dir = 'lineart'
        self.lc_dir = 'lc'
        self.lc_avg_dir = 'lc_avg' if lc_avg == None else lc_avg
        self.pc_ID_dir = 'pm_id'
        self.cp = 'cp'
        self.image_color = 'c'
        self.case_list = os.listdir(os.path.join(mengbao_dir, self.l_dir))
        self.case_list = [case.split(".")[0] for case in self.case_list if case.endswith('.png')]
        self.case_list = [case for case in self.case_list if "Comics" not in case]
        if stage_type == 'train':
            self.case_list = [case for case in self.case_list if int(case.split("_")[-1]) % 5 != 0 and case != '1-2_3-7_658']
        elif stage_type == 'test':
            self.case_list = [case for case in self.case_list if int(case.split("_")[-1]) % 5 == 0 and case != '1-2_3-7_658']
    
        self.stage_type = stage_type


        self.mengbao_pic_len = len(self.case_list)
        
        #########################################################
        # other work
        self.filename_df = pd.read_parquet(other_work_parquet)
        self.other_work_len = len(self.filename_df)
        self.comics_dir = comics_dir
        self.l_dir_comics = 'l'
        self.lc_dir_comics = 'lc'
        self.lc_avg_dir_comics = 'lc_avg' if lc_avg == None else lc_avg
        self.pc_ID_dir_comics = 'person_clothes_ID_map'
        self.cp_comics = 'cp'


    def __len__(self):
        if self.other_work:
            return self.mengbao_pic_len + self.mengbao_pic_len // 3
        else:
            return self.mengbao_pic_len

    def __getitem__(self, index):
        '''
            Every item in the dataset, select one of the person_clothes_id
        '''
        if index < self.mengbao_pic_len:

            # img_lc_path = os.path.join(self.data_dir, self.lc_avg_dir, self.filename_df.iloc[index]['filename'] + '.png')
            img_l_path = os.path.join(self.mengbao_dir, self.l_dir , self.case_list[index] + '.png')

            image_l = Image.open(img_l_path).convert("RGB")

            image = np.array(image_l)
            cp_mask = np.load(os.path.join(self.mengbao_dir, self.cp, self.case_list[index] + '.npy'))
            pc_ID_map = np.load(os.path.join(self.mengbao_dir, self.pc_ID_dir, self.case_list[index] + '.npy'))

            pc_unique_ID = np.unique(pc_ID_map)
            pc_unique_ID = pc_unique_ID[pc_unique_ID != 0]
            # original data code
            pc_ID = np.random.choice(pc_unique_ID)
            # tmp check data code
            # pc_ID_map_backup = copy.deepcopy(pc_ID_map)
            # cp_mask_backup = copy.deepcopy(cp_mask)
            # for pc_ID in pc_unique_ID:

            #     # tmp check data code
            #     pc_ID_map = copy.deepcopy(pc_ID_map_backup)
            #     cp_mask = copy.deepcopy(cp_mask_backup)

            cp_mask_unique = np.unique(cp_mask)
            cp_mask_max = np.max(cp_mask_unique)
            cp_mask[cp_mask != cp_mask_max] = 0
            cp_mask[cp_mask == cp_mask_max] = 1

            pc_ID_map[pc_ID_map != pc_ID] = 0
            pc_ID_map[pc_ID_map == pc_ID] = 1

            pc_ID_map = pc_ID_map - cp_mask

            mask_ID_map = np.stack([pc_ID_map, pc_ID_map, pc_ID_map], axis=2)
            # print(image.shape, mask_ID_map.shape, cp_mask.shape)

            image_cur = np.where(mask_ID_map == 1, image, 255)
            rows, cols = np.where(pc_ID_map == 1)
            # 计算bbox的左上角和右下角
            top_left = (np.min(rows), np.min(cols))
            bottom_right = (np.max(rows), np.max(cols))
            # top_left = (np.min(cols), np.min(rows))
            # bottom_right = (np.max(cols), np.max(rows))
            image_cur = image_cur[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]

            # image_1 = Image.fromarray(image).convert('RGB')
            # image_1.save(f"test_img_3/{self.filename_df.iloc[index]['filename']}.png")
            # pdb.set_trace()
            if self.transform is not None:
                augmentations = self.transform(image=image_cur)
                image_cur = augmentations['image']
                image_cur = image_cur[[2, 1, 0]] # RGB=>BGR


            return image_cur, int(pc_ID) - 1
        #############################################################################
        #############################################################################
        #############################################################################
        else:
            index = index - self.mengbao_pic_len
            if self.stage_type == 'train':
                index = index
            elif self.stage_type == 'test':
                index = self.other_work_len - index - 1

            img_l_path = os.path.join(self.comics_dir, self.l_dir_comics ,self.filename_df.iloc[index]['filename'] + '.png')

            image_l = Image.open(img_l_path).convert("RGB")

            image = np.array(image_l)
            cp_mask = np.load(os.path.join(self.comics_dir, self.cp_comics, self.filename_df.iloc[index]['filename']+'.npy'))
            pc_ID_map = np.load(os.path.join(self.comics_dir, self.pc_ID_dir_comics, self.filename_df.iloc[index]['filename']+'.npy'))

            pc_unique_ID = np.unique(pc_ID_map)
            pc_unique_ID = pc_unique_ID[pc_unique_ID != 0]
            pc_ID = np.random.choice(pc_unique_ID)
            # print(pc_ID)

            cp_mask_unique = np.unique(cp_mask)
            cp_mask_max = np.max(cp_mask_unique)
            cp_mask[cp_mask != cp_mask_max] = 0
            cp_mask[cp_mask == cp_mask_max] = 1

            pc_ID_map[pc_ID_map != pc_ID] = 0
            pc_ID_map[pc_ID_map == pc_ID] = 1

            pc_ID_map = pc_ID_map - cp_mask

            mask_ID_map = np.stack([pc_ID_map, pc_ID_map, pc_ID_map], axis=2)
            # print(image.shape, mask_ID_map.shape, cp_mask.shape)

            image = np.where(mask_ID_map == 1, image, 255)
            rows, cols = np.where(pc_ID_map == 1)
            # 计算bbox的左上角和右下角
            top_left = (np.min(rows), np.min(cols))
            bottom_right = (np.max(rows), np.max(cols))
            # top_left = (np.min(cols), np.min(rows))
            # bottom_right = (np.max(cols), np.max(rows))
            image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]

            # image_1 = Image.fromarray(image).convert('RGB')
            # image_1.save(f"test_img_3/{self.filename_df.iloc[index]['filename']}.png")
            # pdb.set_trace()
            if self.transform is not None:
                augmentations = self.transform(image=image)
                image = augmentations['image']
                image = image[[2, 1, 0]] # RGB=>BGR


            return image, 3


if __name__ == '__main__':
    transform = albumentations.Compose(
    [
        albumentations.Resize(height=448, width=448),
        albumentations.Rotate(limit=35, p=1.0),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.1),
        albumentations.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
    )
    comics_dir = '/data/liuhaoxiang/data_97/comics_local'
    mengbao_dir = '/data/liuhaoxiang/longhao_97/share/mengbao'
    img_filename_parquet = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/data_parquet/2dcomics_train_clearly.parquet'

    # img_test_save_parquet = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/data_parquet/2dcomics_test_clearly.parquet'

    dataset = MengbaoDataset(mengbao_dir, comics_dir, img_filename_parquet, transform=transform)
    # image, pc_ID = dataset[1]
    # pdb.set_trace(
    ## get parquet
    # save_dir = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/data_parquet'
    # time_cost = get_parquet_2dcomics(data_dir, save_dir, type_data='train')
    # _, time_cost = check_parquet_2dcomics('/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/data_parquet/2dcomics_train.parquet')
    # print(time_cost)
    pdb.set_trace()

    for i in tqdm(range(len(dataset))):
        # _, _ = dataset[i]
        _, _ = dataset[i]
    


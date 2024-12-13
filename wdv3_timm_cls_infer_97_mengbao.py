import os
import torch
import torch.optim as optim
import copy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import albumentations
import glob
import time
import pdb
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from wdv3_cls import ComicsDataset, Wdv3Classifier, Wdv3ViTModelOptions, MengbaoDataset, Wdv3Classifier_downstream, FinetuneWdv3ViTModelOptions

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

transform_test = albumentations.Compose(
    [
        albumentations.Resize(height=448, width=448),
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

Comics_person_id_nums = 600

def train_loop(loader, model, optimizer, loss_fn, scaler, device, epoch, batch_size, writer, ckpt_save_path):
    loop = tqdm(loader)
    for batch_idx, (image, targets) in enumerate(loop):
        image = image.to(device)
        targets = targets.to(device)
        # print(targets)
        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(image)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

        # pdb.set_trace()
        if ((batch_idx+1) * batch_size) % 100 == 0:
            writer.add_scalar('Loss/train', loss.item(), (batch_idx+1) * batch_size)
            print(f"Epoch [{epoch}], Step [{(batch_idx+1) * batch_size} ], Loss: {loss.item()}")

 

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, targets in loader:
            image = image.to(device)
            targets = targets.to(device)

            scores = model(image)
            _, predictions = scores.max(1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    print(f"Accuracy: {correct / total * 100:.2f}%")
    model.train()
    return correct / total * 100

def main_cls_4():
    Mengbao_person_id_nums = 4

    data_dir = '/data/liuhaoxiang/data_97/comics_local'
    mengbao_dir = '/data/liuhaoxiang/longhao_97/share/mengbao'
    img_train_parquet = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/data_parquet/2dcomics_train_clearly.parquet'
    img_test_parquet = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/data_parquet/2dcomics_train_clearly.parquet'

    save_root = '/data/liuhaoxiang/data_97/experiments/wdv3_cls_viz/'
    os.makedirs(save_root, exist_ok=True)

    model_name = 'l_mengbao_ft'

    os.makedirs(os.path.join(save_root, model_name), exist_ok=True)

    test_dataset = MengbaoDataset(
        mengbao_dir=mengbao_dir,
        comics_dir=data_dir,
        other_work_parquet=img_test_parquet,
        transform=transform_test,
        stage_type='test'
    )


    model = Wdv3Classifier_downstream(
        FinetuneWdv3ViTModelOptions, 
        num_classes=Mengbao_person_id_nums, 
        local_pretrain_model_num_classes=Comics_person_id_nums
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ckpt_path = '/data/liuhaoxiang/data_97/experiments/wdv3_cls/l_mengbao_ft/model/best_epoch_model.pth'
    if os.path.exists(ckpt_path):
        resume_checkpoint = torch.load(ckpt_path)
        model.load_state_dict(resume_checkpoint['state_dict'])
        # optimizer.load_state_dict(resume_checkpoint['optimizer'])
        print('Successful Load Model')


    for i in range(len(test_dataset)):
        if i < test_dataset.mengbao_pic_len:
            case_name = test_dataset.case_list[i]
        else:
            case_name = test_dataset.filename_df.iloc[i]['filename']

        image, label = test_dataset[i]
        image_npy = image.numpy()
        image_npy = Image.fromarray((image_npy * 255).astype(np.uint8).transpose(1, 2, 0))

        image = image.unsqueeze(0).to(device)
 
        with torch.no_grad():
            prediction = model(image)
            prob = nn.functional.softmax(prediction)
            print("prob:", prob)
            print(int(torch.argmax(prob)))
            print("max prob:", prob.max())
            print(label)
            
        image_npy.save(os.path.join(save_root, model_name, f'{case_name}_{label}_pred_{int(torch.argmax(prob))}_prob_{prob.max()}.jpg'))
        # pdb.set_trace()

def main_test_4():
    Mengbao_person_id_nums = 4
    name_dict = {
        0: 'mengbao', 
        1:'woman',
        2: 'man',
        3: 'other'
    }
    
    path_root = '/data/liuhaoxiang/data_97/experiments/mengbao_real_test'

    img_root = os.path.join(path_root, 'lineart')
    mask_root = os.path.join(path_root, 'pm')
    save_root = os.path.join(path_root, 'cls_result_cls_4')
    os.makedirs(save_root, exist_ok=True)

    img_list = glob.glob(os.path.join(img_root, '*.jpg'))
    img_list = img_list + glob.glob(os.path.join(img_root, '*.JPG'))

    model = Wdv3Classifier_downstream(
        FinetuneWdv3ViTModelOptions, 
        num_classes=Mengbao_person_id_nums, 
        local_pretrain_model_num_classes=Comics_person_id_nums
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # ckpt_path = '/data/liuhaoxiang/data_97/experiments/wdv3_cls/l_mengbao_ft/model/best_epoch_model.pth'
    ckpt_path = '/data/liuhaoxiang/data_97/experiments/wdv3_cls/l_mengbao_ft_1734022866.233696/model/model_epoch_3.pth'
    if os.path.exists(ckpt_path):
        resume_checkpoint = torch.load(ckpt_path)
        model.load_state_dict(resume_checkpoint['state_dict'])
        # optimizer.load_state_dict(resume_checkpoint['optimizer'])
        print('Successful Load Model')

    all_case = 0
    correct_case = 0
    for img in tqdm(img_list):
        # try:
        basename = os.path.basename(img).split('.')[0]
        mask_path = os.path.join(mask_root, f'{basename}.npy')

        # pdb.set_trace()
        mask = np.load(mask_path)
        mask_unique_ID = np.unique(mask)
        mask_unique_ID = mask_unique_ID[mask_unique_ID != 0]
        for pc_ID in mask_unique_ID:
            
            mask_ID_map = copy.deepcopy(mask)

            mask_ID_map[mask_ID_map != pc_ID] = 0
            mask_ID_map[mask_ID_map == pc_ID] = 1

            mask_h, mask_w = mask.shape

            rows, cols = np.where(mask_ID_map == 1)
            top_left = (np.min(rows), np.min(cols))
            bottom_right = (np.max(rows), np.max(cols))
            mask_RGB_map = np.stack([mask_ID_map, mask_ID_map, mask_ID_map], axis=2)

            image = Image.open(img)
            image = image.resize((mask_h, mask_w))
            image_npy = np.array(image)

            image_npy = np.where(mask_RGB_map == 1, image_npy, 255)
            image_npy = image_npy[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]

            augmentations = transform_test(image=image_npy)
            image_tensor = augmentations['image']
            image_tensor = image_tensor[[2, 1, 0]] # RGB=>BGR
            image_tensor = image_tensor.unsqueeze(0).to(device)
            prediction = model(image_tensor)
            prob = nn.functional.softmax(prediction)
            max_prob_index = int(torch.argmax(prob))
            char_name = name_dict[max_prob_index]
            real_name = name_dict[int(pc_ID)-1]
            if char_name == real_name:
                correct_case += 1
            prob = prob[0].cpu().detach().numpy()
            prob_list = [round(float(ele), 2) for ele in prob]
            name_prob_lits = [f'{name_dict[i]}:{prob_list[i]}' for i in range(len(prob_list))]
            
            

            image = Image.fromarray(image_npy)

            image.save(os.path.join(save_root, f'{basename}_real_id_{int(pc_ID)-1}_{real_name}_pred_id_{max_prob_index}_{char_name}_prob_{name_prob_lits}.jpg'))
            all_case += 1
    print(f'all_case: {all_case}')
    print(f'correct_case: {correct_case}')
    print(f'correct_case/all_case: {correct_case}/{all_case}={correct_case/all_case}')
        # except:
        #     basename = os.path.basename(img).split('.')[0]
        #     mask_path = os.path.join(mask_root, f'{basename}.npy')
        #     mask = np.load(mask_path)
        #     print(np.unique(mask))
        #     continue
        #     mask = mask.astype(np.uint8)
        #     mask = np.where(mask == 1, 255, 0)
        #     mask = Image.fromarray(mask).convert('RGB')
        #     mask.save(os.path.join(save_root, f'{basename}_bad_mask.jpg'))

def main_test_3():
    Mengbao_person_id_nums = 3
    name_dict = {
        0: 'mengbao', 
        1:'woman',
        2: 'man',
        3: 'other'
    }
    
    path_root = '/data/liuhaoxiang/data_97/experiments/mengbao_real_test'

    img_root = os.path.join(path_root, 'lineart')
    mask_root = os.path.join(path_root, 'pm')
    save_root = os.path.join(path_root, 'cls_result_cls_3')
    os.makedirs(save_root, exist_ok=True)

    img_list = glob.glob(os.path.join(img_root, '*.jpg'))
    img_list = img_list + glob.glob(os.path.join(img_root, '*.JPG'))

    model = Wdv3Classifier_downstream(
        FinetuneWdv3ViTModelOptions, 
        num_classes=Mengbao_person_id_nums, 
        local_pretrain_model_num_classes=Comics_person_id_nums
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # ckpt_path = '/data/liuhaoxiang/data_97/experiments/wdv3_cls/l_mengbao_ft/model/best_epoch_model.pth'
    # ckpt_path = '/data/liuhaoxiang/data_97/experiments/wdv3_cls/l_mengbao_ft_1734022866.233696/model/model_epoch_0.pth'
    ckpt_path = '/data/liuhaoxiang/data_97/experiments/wdv3_cls/l_mengbao_ft_cls_3_1734025890.0528514/model/model_epoch_5.pth'
    print("ckpt: ", ckpt_path)
    if os.path.exists(ckpt_path):
        resume_checkpoint = torch.load(ckpt_path)
        model.load_state_dict(resume_checkpoint['state_dict'])
        # optimizer.load_state_dict(resume_checkpoint['optimizer'])
        print('Successful Load Model')

    all_case = 0
    correct_case = 0
    for img in tqdm(img_list):
        # try:
        basename = os.path.basename(img).split('.')[0]
        mask_path = os.path.join(mask_root, f'{basename}.npy')

        # pdb.set_trace()
        mask = np.load(mask_path)
        mask_unique_ID = np.unique(mask)
        mask_unique_ID = mask_unique_ID[mask_unique_ID != 0]
        for pc_ID in mask_unique_ID:
            
            mask_ID_map = copy.deepcopy(mask)

            mask_ID_map[mask_ID_map != pc_ID] = 0
            mask_ID_map[mask_ID_map == pc_ID] = 1

            mask_h, mask_w = mask.shape

            rows, cols = np.where(mask_ID_map == 1)
            top_left = (np.min(rows), np.min(cols))
            bottom_right = (np.max(rows), np.max(cols))
            mask_RGB_map = np.stack([mask_ID_map, mask_ID_map, mask_ID_map], axis=2)

            image = Image.open(img)
            image = image.resize((mask_h, mask_w))
            image_npy = np.array(image)

            image_npy = np.where(mask_RGB_map == 1, image_npy, 255)
            image_npy = image_npy[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]

            augmentations = transform_test(image=image_npy)
            image_tensor = augmentations['image']
            image_tensor = image_tensor[[2, 1, 0]] # RGB=>BGR
            image_tensor = image_tensor.unsqueeze(0).to(device)
            prediction = model(image_tensor)
            prob = nn.functional.softmax(prediction)
            max_prob_index = int(torch.argmax(prob))
            char_name = name_dict[max_prob_index]
            real_name = name_dict[int(pc_ID)-1]
            if char_name == real_name:
                correct_case += 1
            prob = prob[0].cpu().detach().numpy()
            prob_list = [round(float(ele), 2) for ele in prob]
            name_prob_lits = [f'{name_dict[i]}:{prob_list[i]}' for i in range(len(prob_list))]
            
            

            image = Image.fromarray(image_npy)

            image.save(os.path.join(save_root, f'{basename}_real_id_{int(pc_ID)-1}_{real_name}_pred_id_{max_prob_index}_{char_name}_prob_{name_prob_lits}.jpg'))
            all_case += 1
    print(f'all_case: {all_case}')
    print(f'correct_case: {correct_case}')
    print(f'correct_case/all_case: {correct_case}/{all_case}={correct_case/all_case}')
        # except:


if __name__ == '__main__':

    main_test_3()



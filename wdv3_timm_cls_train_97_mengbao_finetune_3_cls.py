import os
import torch
import torch.optim as optim
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

Comics_person_id_nums = 600
Mengbao_person_id_nums = 3

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

def main():
    data_dir = '/data/liuhaoxiang/data_97/comics_local'
    mengbao_dir = '/data/liuhaoxiang/longhao_97/share/mengbao'
    img_train_parquet = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/data_parquet/2dcomics_train_clearly.parquet'
    img_test_parquet = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/data_parquet/2dcomics_train_clearly.parquet'

    save_root = '/data/liuhaoxiang/data_97/experiments/wdv3_cls/'

    model_name = f'l_mengbao_ft_cls_3_{time.time()}'

    os.makedirs(os.path.join(save_root, model_name), exist_ok=True)

    train_dataset = MengbaoDataset(
        mengbao_dir=mengbao_dir,
        comics_dir=data_dir,
        other_work_parquet=img_train_parquet,
        transform=transform,
        stage_type='train',
        other_work=False
    )
    # pdb.set_trace()
    train_batch_size = 16
    # pdb.set_trace()
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8
    )

    test_dataset = MengbaoDataset(
        mengbao_dir=mengbao_dir,
        comics_dir=data_dir,
        other_work_parquet=img_test_parquet,
        transform=transform,
        stage_type='test',
        other_work=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    loss_fn = nn.CrossEntropyLoss()

    model = Wdv3Classifier_downstream(
        FinetuneWdv3ViTModelOptions, 
        num_classes=Mengbao_person_id_nums, 
        local_pretrain_model_num_classes=Comics_person_id_nums
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # # resume
    # resume_path = '/data/liuhaoxiang/data_97/experiments/wdv3_cls/l_mengbao_ft/model/best_epoch_model.pth'
    # if os.path.exists(resume_path):
    #     resume_checkpoint = torch.load(resume_path)
    #     model.load_state_dict(resume_checkpoint['state_dict'])
    #     optimizer.load_state_dict(resume_checkpoint['optimizer'])
    #     print('Successful Load Model')
    # pdb.set_trace()
    scaler = torch.amp.GradScaler()

    writer = SummaryWriter(os.path.join(save_root, model_name, 'logger'))
    ckpt_save_path = os.path.join(save_root, model_name, 'model')

    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path)

    max_accuracy = 0

    for epoch in range(20):
        train_loop(train_loader, model, optimizer, loss_fn, scaler, device, epoch, train_batch_size, writer, ckpt_save_path)
        # if epoch % 20 == 0:
        accuracy = check_accuracy(test_loader, model, device=device)

        writer.add_scalar('Accuracy/test', accuracy, epoch)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{ckpt_save_path}/model_epoch_{epoch}.pth")
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            torch.save(checkpoint, f"{ckpt_save_path}/best_epoch_model.pth")

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, f"{ckpt_save_path}/last_epoch_model.pth")

def main_check_data():
    data_dir = '/data/liuhaoxiang/data_97/comics_local'
    mengbao_dir = '/data/liuhaoxiang/longhao_97/share/mengbao'
    img_train_parquet = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/data_parquet/2dcomics_train_clearly.parquet'
    img_test_parquet = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/data_parquet/2dcomics_train_clearly.parquet'

    save_root = '/data/liuhaoxiang/data_97/experiments/wdv3_cls/'
    model_name = f'l_mengbao_ft_{time.time()}'

    os.makedirs(os.path.join(save_root, model_name), exist_ok=True)

    train_dataset = MengbaoDataset(
        mengbao_dir=mengbao_dir,
        comics_dir=data_dir,
        other_work_parquet=img_train_parquet,
        transform=transform,
        stage_type='train'
    )
    # # pdb.set_trace()
    # train_batch_size = 16
    # # pdb.set_trace()
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=train_batch_size,
    #     shuffle=True,
    #     num_workers=8
    # )

    # test_dataset = MengbaoDataset(
    #     mengbao_dir=mengbao_dir,
    #     comics_dir=data_dir,
    #     other_work_parquet=img_test_parquet,
    #     transform=transform,
    #     stage_type='test'
    # )

    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=4
    # )

    # loss_fn = nn.CrossEntropyLoss()

    # model = Wdv3Classifier_downstream(
    #     FinetuneWdv3ViTModelOptions, 
    #     num_classes=Mengbao_person_id_nums, 
    #     local_pretrain_model_num_classes=Comics_person_id_nums
    # )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # # # resume
    # # resume_path = '/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/experiments/wdv3_comics_finetune_line/wdv3_comics_finetune_classifier_checkpoint_epoch_99_iters_10000.pth'
    # # if os.path.exists(resume_path):
    # #     resume_checkpoint = torch.load(resume_path)
    # #     model.load_state_dict(resume_checkpoint['state_dict'])
    # #     optimizer.load_state_dict(resume_checkpoint['optimizer'])
    # #     print('Successful Load Model')
    # scaler = torch.amp.GradScaler()

    # writer = SummaryWriter(os.path.join(save_root, model_name, 'logger'))
    # ckpt_save_path = os.path.join(save_root, model_name, 'model')

    # if not os.path.exists(ckpt_save_path):
    #     os.makedirs(ckpt_save_path)

    # max_accuracy = 0
    bad_case_list = []
    len_train_dataset = train_dataset.mengbao_pic_len
    for epoch in range(1):
        for i in tqdm(range(len_train_dataset)):
            try:
                _, _ = train_dataset[i]
            except:
                name = train_dataset.case_list[i]
                bad_case_list.append(name)
                print(name)
    print(bad_case_list)
    pdb.set_trace()
    #     train_loop(train_loader, model, optimizer, loss_fn, scaler, device, epoch, train_batch_size, writer, ckpt_save_path)
    #     # if epoch % 20 == 0:
    #     accuracy = check_accuracy(test_loader, model, device=device)

    #     writer.add_scalar('Accuracy/test', accuracy, epoch)
    #     checkpoint = {
    #         "state_dict": model.state_dict(),
    #         "optimizer": optimizer.state_dict(),
    #     }
    #     torch.save(checkpoint, f"{ckpt_save_path}/model_epoch_{epoch}.pth")
    #     if accuracy > max_accuracy:
    #         max_accuracy = accuracy
    #         torch.save(checkpoint, f"{ckpt_save_path}/best_epoch_model.pth")

    # checkpoint = {
    #     "state_dict": model.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    # }
    # torch.save(checkpoint, f"{ckpt_save_path}/last_epoch_model.pth")


if __name__ == '__main__':

    main()

import argparse
import os
import math
import logging
import glob
from PIL.Image import RASTERIZE
import PIL.Image as pil_image
import numpy as np
import cv2
from utils import preprocess

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from models.models import Generator, Discriminator
from models.VapSR import vapsr
from models.realgan_v3 import SRVGGNetCompact

from models.loss import VGGLoss, GANLoss
from utils import (
    AverageMeter,
    calc_psnr,
    calc_ssim,
)
from dataset import Dataset
import config as C


# pid : 6323 python3 train.py --train-file /dataset/data/ --eval-file /dataset/test --outputs-dir weights --num-net-epochs 0 --num-gan-epochs 100000 --resume-g pretrained/RealESRGAN_x4plus.pth  --scale 2 --cuda 3 --patch-size 80

def net_trainer(train_dataloader, eval_dataloader, model, pixel_criterion,p_loss, net_optimizer, epoch, best_psnr, scaler, writer, device, args):
    if epoch % 5 == 0:
        os.makedirs(net_pic_out_test+"/epoch{}".format(epoch), exist_ok=True)
        if os.path.isfile(net_test_input):
            paths = [net_test_input]
        else:
            paths = sorted(glob.glob(os.path.join(net_test_input, '*')))

        for idx, path in enumerate(paths):
            imgname, extension = os.path.splitext(os.path.basename(path))
            extension = extension[1:]
            # print('Testing', idx, imgname)
            image = pil_image.open(path).convert('RGB')
            image_width = (image.width //scale) *scale
            image_height = (image.height // scale) * scale
            
            #lr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
            lr = image
            #bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
            #bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
            
            lr = preprocess(lr).cuda()
            #bic = preprocess(bicubic).to(device)
            generator.eval()
            with torch.no_grad():
                preds = generator(lr)
            preds = preds.mul(255.0).cpu().numpy().squeeze(0)
            
            output = np.array(preds).transpose([1,2,0])
            output = np.clip(output, 0.0, 255.0).astype(np.uint8)
            output = pil_image.fromarray(output)
            save_path = os.path.join(net_pic_out_test+"/epoch{}".format(epoch), f'{imgname}.{extension}')
            output.save(save_path)

    model.train()
    losses = AverageMeter(name="PSNR Loss", fmt=":.6f")
    psnr = AverageMeter(name="PSNR", fmt=":.6f")
    print("开始进行realesrnet的训练===================================")
    """  开始训练纪元 """
    for i, (lr, hr) in enumerate(train_dataloader):
        lr = lr.to(device)
        hr = hr.to(device)

        net_optimizer.zero_grad()

        with amp.autocast():
            preds = model(lr)
            loss = pixel_criterion(preds, hr)+0.1*p_loss(preds, hr)
            #loss = pixel_criterion(preds, hr)

        if i == 0:
            vutils.save_image(lr.detach(), os.path.join(net_pic_out_lr, f"LR_{epoch}.jpg"))
            vutils.save_image(hr.detach(), os.path.join(net_pic_out_hr, f"HR_{epoch}.jpg"))
            vutils.save_image(preds.detach(), os.path.join(net_pic_out_pre, f"preds_{epoch}.jpg"))
            print("epoch{}  loss:{}".format(epoch,loss))
        
        """缩放器更新"""
        scaler.scale(loss).backward()
        scaler.step(net_optimizer)
        scaler.update()

        """ Loss 更新 """
        losses.update(loss.item(), len(lr))
        #print("net loss is =========",loss)
    
    """ 1 epoch 每个张量板 更新 """
    writer.add_scalar('L1Loss/train', losses.avg, epoch)

    """  开始测试纪元 """
    model.eval()
    for i, (lr, hr) in enumerate(eval_dataloader):
        lr = lr.to(device)
        hr = hr.to(device)
        with torch.no_grad():
            preds = model(lr)
        psnr.update(calc_psnr(preds, hr), len(lr))

    """ 1 epoch 每次张量板更新 """
    writer.add_scalar('psnr/test', psnr.avg, epoch)

    if psnr.avg > best_psnr:
        best_psnr = psnr.avg
        torch.save(
            model, os.path.join(net_ckpt, 'best.pth')
        )

    if epoch % 2 == 0:
        torch.save(
        model, os.path.join(net_ckpt, "net_epoch_{}.pth".format(epoch))
        )



if __name__ == '__main__':
    """ 日志设置 """
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

    """ Argparse 설정 """
    parser = argparse.ArgumentParser()
    """数据参数设置"""
    train_file = C.quanju_peizhi['train_file']
    eval_file = C.quanju_peizhi['eval_file']
    outputs_dir = C.quanju_peizhi['outputs_dir']

    """model args setup"""
    scale = C.quanju_peizhi['scale']
    print(scale)

    """ net model args setup"""
    num_net_epochs = C.net_peizhi['num_net_epochs']
    resume_net = C.net_peizhi['resume_net']
    psnr_lr = C.net_peizhi['psnr_lr']
    net_pic_out = C.net_peizhi['net_pic_out']
    net_pic_out_pre = C.net_peizhi['net_pic_out_pre']
    net_pic_out_lr = C.net_peizhi['net_pic_out_lr']
    net_pic_out_hr = C.net_peizhi['net_pic_out_hr']
    net_pic_out_test = C.net_peizhi['net_pic_out_test']
    net_test_input = C.net_peizhi['net_test_input']

    net_ckpt = C.net_peizhi['net_ckpt']

    """etc args setup"""
    batch_size = C.quanju_peizhi['batch_size_net']
    patch_size = C.quanju_peizhi['patch_size']
    num_workers = C.quanju_peizhi['num_workers']
    seed = C.quanju_peizhi['seed']
    cuda = C.quanju_peizhi['cuda']
    #parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()
    
    """ 设置路径以保存权重 """ 
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    #net 文件夹
    if not os.path.exists(net_pic_out):
        os.makedirs(net_pic_out)
    if not os.path.exists(net_pic_out_hr):
        os.makedirs(net_pic_out_hr)
    if not os.path.exists(net_pic_out_lr):
        os.makedirs(net_pic_out_lr)
    if not os.path.exists(net_pic_out_pre):
        os.makedirs(net_pic_out_pre)
    
    
    """ TensorBoard 设置 """
    writer = SummaryWriter(outputs_dir)

    """ GPU 디바이스 설정 """
    cudnn.benchmark = True
    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
    
    """ Torch Seed 설정 """
    torch.manual_seed(seed)

    """ RealESRGAN psnr 모델 설정 """
    #generator = Generator(scale).to(device)
    #generator = SRVGGNetCompact().to(device)
    #generator = torch.load(resume_net).to(device)
    generator = vapsr().to(device)
    print(psnr_lr)
    print(generator)

    pixel_criterion = nn.L1Loss().to(device)
    p_loss = VGGLoss().to(device)
    net_optimizer = torch.optim.Adam(generator.parameters(), psnr_lr, (0.9, 0.99))
    interval_epoch = math.ceil(num_net_epochs // 8)
    epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]
    net_scheduler = torch.optim.lr_scheduler.MultiStepLR(net_optimizer, milestones=epoch_indices, gamma=0.5)
    scaler = amp.GradScaler()

    total_net_epoch = num_net_epochs
    start_net_epoch = 0
    best_psnr = 0

    """ RealESNet 체크포인트 weight 불러오기 """
#    if os.path.exists(net_ckpt):
#        checkpoint = torch.load(resume_net)
#        generator.load_state_dict(checkpoint['model_state_dict'])
#        net_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#        start_net_epoch = checkpoint['epoch'] + 1
#        loss = checkpoint['loss']
#        best_psnr = checkpoint['best_psnr']
    if not os.path.exists(net_ckpt):
        os.makedirs(net_ckpt)
    """ RealESRNet 로그 인포 프린트 하기 """


    """ 数据集和数据集设置 """
    train_dataset = Dataset(train_file, patch_size, scale)
    train_dataloader = DataLoader(
                            dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True
                        )
    eval_dataset = Dataset(eval_file, patch_size, scale)
    eval_dataloader = DataLoader(
                                dataset=eval_dataset, 
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True
                                )
    print(start_net_epoch,"=============")
    """NET Training"""
    for epoch in range(start_net_epoch, total_net_epoch):
        net_trainer(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, model=generator, pixel_criterion=pixel_criterion, p_loss=p_loss,net_optimizer=net_optimizer, epoch=epoch, best_psnr=best_psnr, scaler=scaler, writer=writer, device=device, args=args)
        net_scheduler.step()
        

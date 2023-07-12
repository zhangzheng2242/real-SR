import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import os
import math
import logging

from PIL.Image import RASTERIZE
import PIL.Image as pil_image
import numpy as np

import argparse
import glob
import cv2
from utils import preprocess

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
#from models.VapSR import vapsr
from models.realgan_v3 import SRVGGNetCompact

from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
from models.models import Generator, Discriminator
from models.loss import VGGLoss, GANLoss
from utils import (
    AverageMeter,
    calc_psnr,
    calc_ssim,
)
from dataset import Dataset
import config as C


def gan_trainer(train_dataloader, eval_dataloader, generator, discriminator, pixel_criterion, content_criterion, adversarial_criterion, generator_optimizer, discriminator_optimizer, epoch, best_psnr, scaler, writer, args):
    generator.train()
    discriminator.train()

    """ 损失平均计设置 """
    d_losses = AverageMeter(name="D Loss", fmt=":.6f")
    g_losses = AverageMeter(name="G Loss", fmt=":.6f")
    pixel_losses = AverageMeter(name="Pixel Loss", fmt=":6.4f")
    content_losses = AverageMeter(name="Content Loss", fmt=":6.4f")
    adversarial_losses = AverageMeter(name="adversarial losses", fmt=":6.4f")

    """ 设置模型评估测量 """
    psnr = AverageMeter(name="PSNR", fmt=":.6f")
    #ssim = AverageMeter(name="SSIM", fmt=":.6f")
    print("开始进行realesrgan的训练===+++++++++++++++++++++++=====================")
    print("==========开始进行realesrgan的第",epoch,"轮训练=============")

    if epoch % 5 == 0:
        os.makedirs(gan_pic_out_test+"/epoch{}".format(epoch), exist_ok=True)
        if os.path.isfile(gan_test_input):
            paths = [gan_test_input]
        else:
            paths = sorted(glob.glob(os.path.join(gan_test_input, '*')))

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
            save_path = os.path.join(gan_pic_out_test+"/epoch{}".format(epoch), f'{imgname}.{extension}')
            output.save(save_path)


    """  开始训练纪元 """
    for i, (lr, hr) in enumerate(train_dataloader):
        """LR & HR 디바이스 설정"""
        lr = lr.cuda()
        hr = hr.cuda()

        """ 标识符优化初始化 """
        discriminator_optimizer.zero_grad()

        with amp.autocast():
            """推理"""
            preds = generator(lr)
            """ 计算通过标识符后的损失 """
            real_output = discriminator(hr)
            d_loss_real = adversarial_criterion(real_output, True)

            fake_output = discriminator(preds.detach())
            d_loss_fake = adversarial_criterion(fake_output, False)

            d_loss = (d_loss_real + d_loss_fake) / 2
        
        """ 体重更新 """
        scaler.scale(d_loss).backward()
        scaler.step(discriminator_optimizer)
        scaler.update()

        """ 构造函数优化初始化 """
        generator_optimizer.zero_grad()

        with amp.autocast():
            """推理"""
            preds = generator(lr)
            """ 计算通过标识符后的损失 """
            real_output = discriminator(hr.detach())
            fake_output = discriminator(preds)
            pixel_loss = pixel_criterion(preds, hr.detach())
            content_loss = content_criterion(preds, hr.detach())
            adversarial_loss = adversarial_criterion(fake_output, True)
            g_loss = 1 * pixel_loss + 1 * content_loss + 0.1 * adversarial_loss
        # print("g_loss is ==============",g_loss,"d_loss is ==============",d_loss)
        """ 每 1 个 epoch 检查一次测试图像 """
        if i == 0:
            vutils.save_image(
                lr.detach(), os.path.join(gan_pic_out_lr, f"LR_{epoch}.jpg")
            )
            vutils.save_image(
                hr.detach(), os.path.join(gan_pic_out_hr, f"HR_{epoch}.jpg")
            )
            vutils.save_image(
                preds.detach(), os.path.join(gan_pic_out_pre, f"preds_{epoch}.jpg")
            )
            print("epoch{}  g_loss:{} === d_loss:{}".format(epoch,g_loss,d_loss))

        """ 体重更新 """
        scaler.scale(g_loss).backward()
        scaler.step(generator_optimizer)
        scaler.update()

        """ 构造函数初始化 """
        generator.zero_grad()

        """ 损失更新 """
        d_losses.update(d_loss.item(), lr.size(0))
        g_losses.update(g_loss.item(), lr.size(0))
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))

    """ 调度程序更新 """
    discriminator_scheduler.step()
    generator_scheduler.step()

    """ Tensorboard 每 1 个 epoch 更新一次 """
    writer.add_scalar("d_Loss/train", d_losses.avg, epoch)
    writer.add_scalar("g_Loss/train", g_losses.avg, epoch)
    writer.add_scalar("pixel_losses/train", pixel_losses.avg, epoch)
    writer.add_scalar("adversarial_losses/train", content_losses.avg, epoch)
    writer.add_scalar("adversarial_losses/train", adversarial_losses.avg, epoch)

    """  开始测试纪元 """
    generator.eval()
    with torch.no_grad():
        for i, (lr, hr) in enumerate(eval_dataloader):
            lr = lr.cuda()
            hr = hr.cuda()
            preds = generator(lr)
            psnr.update(calc_psnr(preds, hr), len(lr))
            #hr = hr.to(torch.float16)
            #ssim.update(calc_ssim(preds, hr).mean(), len(lr))
  
    """ Tensorboard 每 1 个 epoch 更新一次 """
    writer.add_scalar("psnr/test", psnr.avg, epoch)
    #writer.add_scalar("ssim/test", ssim.avg, epoch)

    """  保存最佳模型 """
    print("psnr:", psnr.avg, epoch)
    #print("ssim:", ssim.avg, epoch)

    if psnr.avg > best_psnr:
        best_psnr = psnr.avg
        # torch.save(
        #     generator.state_dict(), os.path.join(gan_ckpt_g, "best_g.pth")
        # )
        torch.save(
        generator, os.path.join(gan_ckpt_g, "best_g_rrdb12_50%.pth")
        )
        torch.save(
        discriminator, os.path.join(gan_ckpt_d, "best_d_rrdb6_70%.pth")
        )

    """ Epoch 1000번에 1번 저장 """
    




    if epoch % 5 == 0:
        """Discriminator 모델 저장""" 
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": discriminator.state_dict(),
                "optimizer_state_dict": discriminator_optimizer.state_dict(),
            },
            os.path.join(gan_ckpt_d, "d_epoch_{}.pth".format(epoch)),
        )

        """ Generator 모델 저장 """
        torch.save(
        generator, os.path.join(gan_ckpt_g, "g_epoch_{}.pth".format(epoch))
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

    """ net model args setup"""
    num_net_epochs = C.net_peizhi['num_net_epochs']
    resume_net = C.net_peizhi['resume_net']
    psnr_lr = C.net_peizhi['psnr_lr']

    """ GAN model args setup"""
    num_gan_epochs = C.gan_peizhi['num_gan_epochs']
    resume_g = C.gan_peizhi['resume_g']
    resume_d = C.gan_peizhi['resume_d']
    gan_lr = C.gan_peizhi['gan_lr']
    gan_pic_out = C.gan_peizhi['gan_pic_out']
    gan_pic_out_pre = C.gan_peizhi['gan_pic_out_pre']
    gan_pic_out_lr = C.gan_peizhi['gan_pic_out_lr']
    gan_pic_out_hr = C.gan_peizhi['gan_pic_out_hr']
    gan_pic_out_test =C.gan_peizhi['gan_test'] 
    gan_test_input = C.gan_peizhi['test_input'] 
    gan_ckpt_g = C.gan_peizhi['gan_ckpt_g']
    gan_ckpt_d = C.gan_peizhi['gan_ckpt_d']
    prune_g = C.gan_peizhi['prune_g']  
    net_weight = C.gan_peizhi['net_weight'] 

    """etc args setup"""
    batch_size = C.quanju_peizhi['batch_size_gan']
    #parser.add_argument('--batch-size', type=int, default=4)
    patch_size = C.quanju_peizhi['patch_size']
    #parser.add_argument('--patch-size', type=int, default=400)
    num_workers = C.quanju_peizhi['num_workers']
    #parser.add_argument('--num-workers', type=int, default=8)
    seed = C.quanju_peizhi['seed']
    #parser.add_argument('--seed', type=int, default=123)
    #cuda = C.quanju_peizhi['cuda']
    #parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()
    
    """ 设置路径以保存权重 """ 
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    
    
    #gan 文件夹
    if not os.path.exists(gan_pic_out):
        os.makedirs(gan_pic_out)
    if not os.path.exists(gan_pic_out_hr):
        os.makedirs(gan_pic_out_hr)
    if not os.path.exists(gan_pic_out_lr):
        os.makedirs(gan_pic_out_lr)
    if not os.path.exists(gan_pic_out_pre):
        os.makedirs(gan_pic_out_pre)
    

    """ TensorBoard 设置 """
    writer = SummaryWriter(outputs_dir)

    """ GPU 디바이스 설정 """
    cudnn.benchmark = True
    #device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
    """ Torch Seed 설정 """
    torch.manual_seed(seed)

    """ RealESRGAN psnr 모델 설정 """
    #generator = Generator(scale).cuda()  #实例化生成器网络
    #generator = vapsr().cuda()
    
    #generator.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(net_weight).items()})
    #generator.load_state_dict(torch.load(net_weight, map_location='cuda:0'))
    #generator = torch.load(prune_g).cuda()
    generator = torch.load(net_weight).cuda()
    print(generator)


    """ 加载 RealESNet 鉴别器检查点权重 """
    discriminator = Discriminator().cuda()   #实例化判别器


    
    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    pixel_criterion = nn.L1Loss().cuda()
    scaler = amp.GradScaler()

    total_net_epoch = num_net_epochs
    start_net_epoch = 0
    best_psnr = 0


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


    total_gan_epoch = num_gan_epochs
    start_gan_epoch = 0
    # best_ssim = 0
    best_psnr = 0

    content_criterion = VGGLoss().cuda()
    adversarial_criterion = GANLoss().cuda()

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=gan_lr, betas=(0.9, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=gan_lr, betas=(0.9, 0.999))

    interval_epoch = math.ceil(num_gan_epochs // 8)
    epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]
    discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_optimizer, milestones=epoch_indices, gamma=0.5)
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(generator_optimizer, milestones=epoch_indices, gamma=0.5)



    """ 加载检查点权重 """
    if os.path.exists(resume_g) :
        checkpoint_g = torch.load(resume_g)
        generator.load_state_dict(checkpoint_g['model_state_dict'])
        start_gan_epoch = checkpoint_g['epoch'] + 1
        generator_optimizer.load_state_dict(checkpoint_g['optimizer_state_dict'])

    if os.path.exists(resume_d):
        """ resume discriminator """
        checkpoint_d = torch.load(resume_d)
        discriminator.load_state_dict(checkpoint_d['model_state_dict'],False)
        discriminator_optimizer.load_state_dict(checkpoint_d['optimizer_state_dict'])


    if not os.path.exists(gan_ckpt_g):
        os.makedirs(gan_ckpt_g)
    if not os.path.exists(gan_ckpt_d):
        os.makedirs(gan_ckpt_d)
    """GAN Training"""
    for epoch in range(start_gan_epoch, total_gan_epoch):
        gan_trainer(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, generator=generator, discriminator=discriminator, pixel_criterion=pixel_criterion, content_criterion=content_criterion, adversarial_criterion=adversarial_criterion, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, epoch=epoch, best_psnr=best_psnr, scaler=scaler, writer=writer,  args=args)
        discriminator_scheduler.step()
        generator_scheduler.step()


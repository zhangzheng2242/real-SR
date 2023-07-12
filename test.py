import argparse
import os
import glob
import cv2
import torch
import time

import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from torch import nn
from models.models import Generator
from models.VapSR import vapsr
from models.realgan_v3 import SRVGGNetCompact

from utils import preprocess
import time
# python test.py --weights-file weights/LDSR_DENSE_light_DEGRADE_x4_2_14x256/best.pth --image-file examples/sweden_samples/192899_061337_updates.jpg --scale 4 --merge
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='/home/stc/code/prune/weights/vapsrS-x2/vapsrS-x2.pth')
    # parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, default='/home/stc/code/prune/input-test/1080p', help='Input image or folder')
    # /home/dell/chaofen/rval_11_30
    # /home/stc/code/prune/input-test/MTF
    #/home/stc/code/prune/input/4k/4K1

    parser.add_argument('-o', '--output', type=str, default='/home/stc/code/prune/results/shijian', help='Output folder')

#/home/stc/code/prune/results/shijian

    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument(
        '--ext',
        type=str,

        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    args = parser.parse_args()

    #cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')  

    model = torch.load(args.weights_file).to(device)

   
    #model = Generator(scale=args.scale).to(device)

    #model = vapsr().to(device)
    #model = SRVGGNetCompact().to(device)


    # solution 2
    #model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.weights_file).items()}, strict=True)
    #state_dict = torch.load(args.weights_file)
    #model.load_state_dict(state_dict['params'], strict=True)
    #model.load_state_dict(state_dict, strict=True)



    # state_dict = model.state_dict()
    # try:
    #     for n, p in torch.load(args.weights_file,map_location=device).items():
    #         if n in state_dict.keys():
    #             state_dict[n].copy_(p)
    #         else:
    #             raise KeyError(n)
    # except:
    #     for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage)['model_state_dict'].items():
    #         if n in state_dict.keys():
    #             state_dict[n].copy_(p)
    #         else:
    #             raise KeyError(n)


    model.eval()


    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.output+"/bicubic", exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        extension = extension[1:]
        print('Testing', idx, imgname)
        image = pil_image.open(path).convert('RGB')
        #image_width = (image.width // args.scale) * args.scale
        #image_height = (image.height // args.scale) * args.scale
        
        #lr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = image

        #双线性差值保存图像
        #bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        #bicubic.save(os.path.join(args.output+"/bicubic", f'{imgname}.{extension}'))
        
        lr = preprocess(lr).to(device)
        #bic = preprocess(bicubic).to(device)


        with torch.no_grad():
            start_time = time.time()
            preds = model(lr)
            end_time = time.time()
            execution_time = end_time - start_time
            print("执行时间：", execution_time, "秒")

        preds = preds.mul(255.0).cpu().numpy().squeeze(0)

        #traced_net = torch.jit.trace(model, lr)
        #traced_net.save("./weights/vapsr-x2net_cpu.pt")
        
        output = np.array(preds).transpose([1,2,0])
        output = np.clip(output, 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        save_path = os.path.join(args.output, f'{imgname}.{extension}')
        output.save(save_path)


import torch
from models.realgan_v3 import SRVGGNetCompact
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='/home/stc/code/prune/weights/SRVGGNet/realesr-general-x4v3.pth')
    args = parser.parse_args()

    #model = torch.load(args.weights_file,map_location=torch.device('cpu'))
    model = SRVGGNetCompact()
    #model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.weights_file,map_location=torch.device('cpu')).items()}, strict=True) 
    state_dict = torch.load(args.weights_file, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['params'], strict=True)
    model.eval()
    lr = torch.rand(1, 3, 40, 40) 
    #traced_net = torch.jit.script(model)
    traced_net = torch.jit.trace(model, lr)
    traced_net.save("./weights/SRVGGNet2-x4.pt")
    
import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import NoiseNetwork
from utils import *
from torchvision import transforms


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.



def main():
    noisy_file = r'C:\Users\Gingi\Desktop\winter2021\805\FINAL_DEMO1\data\denoised_images'
    out_file = r'C:\Users\Gingi\Desktop\winter2021\805\FINAL_DEMO1\data\noised_images'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Build model
    print('Loading model ...\n')
    net = NoiseNetwork()
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    # print(glob.glob(os.path.join('data', opt.test_data, '*.png')))
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    for f in files_source:
        print('read ' + f)
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))

        # Img2 = np.expand_dims(Img, 0)
        # Img2 = np.expand_dims(Img2, 1)

        ISource = torch.Tensor(Img)
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        INoisy = ISource + noise
        # =========================================================
        n_np = INoisy.cpu().numpy()
        I_n = n_np * 255

        print("Generated noised Files")
        cv2.imwrite(noisy_file + f.split("\\")[-1], I_n)
        # =========================================================
        Img2 = np.expand_dims(Img, 0)
        Img2 = np.expand_dims(Img2, 1)

        ISource = torch.Tensor(Img2)
        INoisy = ISource + torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        #print(INoisy)

        ISource, INoisy = ISource.to(device), INoisy.to(device)
        with torch.no_grad(): # this can save much memory
            de = model(INoisy)
            print(de)
            Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
            #print(Out)
            # =========================================================
            o_np = Out.cpu().clone().numpy()
            print("onp", o_np)
            I_o = o_np.squeeze(0)
            m, n = I_o.shape[1], I_o.shape[2]
            I_o = I_o.reshape((m, n, 1))

            print("Generated Denoised Files")
            cv2.imwrite(out_file + f.split("\\")[-1], I_o * 255)
        # =========================================================
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()

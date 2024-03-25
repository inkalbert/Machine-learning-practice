import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.tensorboard as tb
import torch.optim as optim
class discriminator(nn.Module):
    def __init__(self, input_size, feature_dim):
        super(discriminator, self).__init__()
        self.stuff=nn.Sequential(
            nn.Conv2d(input_size,feature_dim,padding=1,stride=2,kernel_size=4),
            nn.LeakyReLU(0.2),
            self._block(feature_dim,feature_dim*2,4,2,1),
            self._block(feature_dim*2, feature_dim * 4,4, 2, 1),
            self._block(feature_dim*4, feature_dim * 8, 4, 2, 1),
            nn.Conv2d(feature_dim*8,1,kernel_size=4 ,padding=0,stride=2),
            nn.Sigmoid()

        )
    def _block(self,inchannel,outchannel,kernelsize,stride,padding):
        return nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernelsize,stride,padding,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(0.2)

        )
    def forward(self, x):
        return self.stuff(x)
class generator(nn.Module):
    def __init__(self,z_dim,img_dim,feature_dim):
        super(generator, self).__init__()
        self.stuff=nn.Sequential(
            self._block(z_dim,feature_dim*16,4,1,0),
            self._block(feature_dim*16,feature_dim*8,4,2,1),
            self._block(feature_dim*8,feature_dim*4,4,2,1),
            self._block(feature_dim*4,feature_dim*2,4,2,1),
            nn.ConvTranspose2d(feature_dim*2,img_dim,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
    def _block(self,inchannel,outchannel,kernelsize,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(inchannel,outchannel,kernelsize,stride,padding,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(0.2)

        )
    def forward(self,x):
        return self.stuff(x)

def initialize_weights(m):
    for m in m.modules():
        if isinstance(m, (nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)
def test():
   N,inchannel,H,W=8,1,64,64
   z_dim=100
   x=torch.randn((N,inchannel,H,W))
   dis=discriminator(inchannel,8)
   assert dis(x).shape == (N,1,1,1)
   gen=generator(z_dim,inchannel,8)
   z=torch.randn((N,z_dim,1,1))
   assert gen(z).shape == (N,inchannel,H,W)
   print("success")
#n batch_size
#inchannel color channel(1)
#H height of feature space
#W weight of feature space
test()

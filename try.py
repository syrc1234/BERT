#判断是否安装了cuda
#import torch
#print(torch.cuda.is_available())  #返回True则说明已经安装了cuda
#判断是否安装了cuDNN
#from torch.backends import  cudnn
#print(cudnn.is_available())  #返回True则说明已经安装了cuDNN

from model import BertClassifier, Discriminator
from torch.utils.tensorboard import SummaryWriter
import torch
classifier = BertClassifier()
discriminator = Discriminator()
data = torch.randn([1, 768])
writer = SummaryWriter("classshape")
writer.add_graph(classifier, data)
writer.close()
writer = SummaryWriter("dishape")
writer.add_graph(discriminator, data)
writer.close()
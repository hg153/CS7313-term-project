from __future__ import print_function
import pdb
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from models import hsm
from utils import logger

"""import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')"""


torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser(description='HSM-Net')
parser.add_argument('--maxdisp', type=int ,default=384,
                    help='maxium disparity')
parser.add_argument('--logname', default='result_test',
                    help='log name')
parser.add_argument('--database', default='./dataset',
                    help='data path')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batchsize', type=int, default=4,
                    help='samples per batch')
parser.add_argument('--loadmodel', default='./result_test/intermediate.tar',
                    help='weights path')
parser.add_argument('--savemodel', default='./',
                    help='save path')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
torch.manual_seed(args.seed)

model = hsm(args.maxdisp,clean=False,level=1)
#model = nn.DataParallel(model)
model.cuda()

# load model
if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if ('disp' not in k) }
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

def _init_fn(worker_id):
    np.random.seed()
    random.seed()
torch.manual_seed(args.seed)  # set again
torch.cuda.manual_seed(args.seed)


from dataloader import KITTIloader2015 as lk15
from dataloader import KITTIloader2012 as lk12
from dataloader import MiddleburyLoader as DA

batch_size = args.batchsize
scale_factor = args.maxdisp / 384. # controls training resolution


all_left_img, all_right_img, all_left_disp,_,_,_ = lk15.dataloader('%s/data_scene_flow/training/'%args.database,typ='train') # change to trainval when finetuning on KITTI
loader_kitti15 = DA.myImageFloder(all_left_img,all_right_img,all_left_disp, rand_scale=[0.9,2.4*scale_factor], order=0)
all_left_img, all_right_img, all_left_disp = lk12.dataloader('%s/KITTI2012/training/'%args.database)
loader_kitti12 = DA.myImageFloder(all_left_img,all_right_img,all_left_disp, rand_scale=[0.9,2.4*scale_factor], order=0)


data_inuse = torch.utils.data.ConcatDataset([loader_kitti15] + [loader_kitti12]*80 )

TrainImgLoader = torch.utils.data.DataLoader(
         data_inuse, 
         batch_size= batch_size, shuffle= True, num_workers=0, drop_last=True, worker_init_fn=_init_fn)

print('%d batches per epoch'%(len(data_inuse)//batch_size))



def train(imgL,imgR,disp_L):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        disp_L = Variable(torch.FloatTensor(disp_L))

        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        #---------
        mask = (disp_true > 0) & (disp_true < args.maxdisp)
        mask.detach_()
        #----

        optimizer.zero_grad()
        stacked,entropy = model(imgL,imgR)
        loss = (64./85)*F.smooth_l1_loss(stacked[0][mask], disp_true[mask], size_average=True) + \
               (16./85)*F.smooth_l1_loss(stacked[1][mask], disp_true[mask], size_average=True) + \
                (4./85)*F.smooth_l1_loss(stacked[2][mask], disp_true[mask], size_average=True) + \
                (1./85)*F.smooth_l1_loss(stacked[3][mask], disp_true[mask], size_average=True)
        loss.backward()
        optimizer.step()
        vis = {}
        vis['output3'] = stacked[0].detach().cpu().numpy()
        vis['output4'] = stacked[1].detach().cpu().numpy()
        vis['output5'] = stacked[2].detach().cpu().numpy()
        vis['output6'] = stacked[3].detach().cpu().numpy()
        vis['entropy'] = entropy.detach().cpu().numpy()
        lossvalue = loss.data

        del stacked
        del loss
        return lossvalue,vis


def adjust_learning_rate(optimizer, epoch):
    if epoch <= args.epochs - 1:
        lr = 1e-3
    else:
        lr = 1e-4
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    log = logger.Logger(args.savemodel, name=args.logname)
    total_iters = 0

    for epoch in range(1, args.epochs+1):
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)
        
        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):

            start_time = time.time() 
            loss,vis = train(imgL_crop,imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss

            if total_iters %10 == 0:
                log.scalar_summary('train/loss_batch',loss, total_iters)
            if total_iters %100 == 0:
                log.image_summary4('train/left',imgL_crop[0:1],total_iters)
                log.image_summary4('train/right',imgR_crop[0:1],total_iters)
                log.image_summary3('train/gt0',disp_crop_L[0:1],total_iters)
                log.image_summary2('train/entropy',vis['entropy'][0:1],total_iters)
                log.histo_summary('train/disparity_hist',vis['output3'], total_iters)
                log.histo_summary('train/gt_hist',np.asarray(disp_crop_L), total_iters)
                log.image_summary2('train/output3',vis['output3'][0:1],total_iters)
                log.image_summary2('train/output4',vis['output4'][0:1],total_iters)
                log.image_summary2('train/output5',vis['output5'][0:1],total_iters)
                log.image_summary2('train/output6',vis['output6'][0:1],total_iters)
                    
            total_iters += 1

            if (total_iters + 1)%2000==0:
                #SAVE
                savefilename = args.savemodel+'/'+args.logname+'/finetune_'+str(total_iters)+'.tar'
                torch.save({
                    'iters': total_iters,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
                }, savefilename)

        log.scalar_summary('train/loss',total_train_loss/len(TrainImgLoader), epoch)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

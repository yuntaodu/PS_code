import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
import loss
import torch.nn.functional as F
from utils import *
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([
          transforms.Resize((resize_size, resize_size)),
          transforms.CenterCrop(crop_size),
          transforms.ToTensor(),
          torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders


def test_target(args, zz = ''):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/target_F_' + str(zz) + '.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/target_B_' + str(zz) + '.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C_' + str(zz) + '.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, args.dset=="visda17")
    log_str = '\nZz: {}, Task: {}, Accuracy = {:.2f}%'.format(zz, args.name, acc) + '\n' + str(acc_list)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
 

def train_target_ps(args, zz=''):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier,
                                   feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()
    oldC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()
    netD = network.discriminator().cuda()

    args.modelpath = args.output_dir_src + '/source_F_' + str(zz) + '.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B_' + str(zz) + '.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C_' + str(zz) + '.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    oldC.load_state_dict(torch.load(args.modelpath))
    oldC.eval()
    netC.train()
    for k, v in oldC.named_parameters():
        v.requires_grad = False

    param_group = []
    param_group_c = []
    param_group_d = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.01}]#0.1
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr *.1}]# 1
    for k, v in netD.named_parameters():
        param_group_d += [{'params': v, 'lr': args.lr *.1}]# 1
    for k, v in netC.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr * .1}]  #1
    optimizer = optim.SGD(param_group,
                          momentum=0.9,
                          weight_decay=5e-4,
                          nesterov=True)
    optimizer_c = optim.SGD(param_group_c,
                            momentum=0.9,
                            weight_decay=5e-4,
                            nesterov=True)
    optimizer_d = optim.SGD(param_group_d,
                            momentum=0.9,
                            weight_decay=5e-4,
                            nesterov=True)
    
    acc_init = 0
    netF.train()
    netB.train()
    netD.train()

    change = 0
    iter_num = 0
    iter_target = iter(dset_loaders["target"])
    interval_iter = ((args.max_epoch) * len(dset_loaders["target"])) // args.interval
    interval_iter //= 15
    while iter_num < (args.max_epoch) * len(dset_loaders["target"]):
        try:
            inputs_test, _, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_target.next()
        if inputs_test.size(0) == 1:
            continue


        if iter_num % interval_iter == 0:
            netF.eval()
            netB.eval()
            pse_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            pse_label = torch.from_numpy(pse_label).cuda()
            netF.train()
            netB.train()


        iter_num += 1
        inputs_test = inputs_test.cuda()
        batch_size=inputs_test.shape[0]

        if True:
            total_loss = 0
            features_test = netB(netF(inputs_test))
            outputs_test = netC(features_test)
            outputs_test_old = oldC(features_test)

            softmax_out = nn.Softmax(dim=1)(outputs_test)
            softmax_out_old = nn.Softmax(dim=1)(outputs_test_old)
            
           
            entropy_old = Entropy(softmax_out_old)
            label = softmax_out_old.argmax(axis=1)
            indx = [i for i in range(batch_size)]
            indx1 = []
            indx2 = []
            for i in range(args.class_num):
                entropy_sub = entropy_old[label == i]
                if len(entropy_sub) == 0:
                    continue
                indx_sub = entropy_sub.topk(1, largest=False)[-1]
                for j in range(len(indx_sub)):
                    indx1.append(indx_sub[j].cpu().detach().tolist())
            for i in range(batch_size):
                if indx[i] in indx1:
                    continue
                else:
                    indx2.append(indx[i])
            indx1 = torch.tensor(indx1).cuda()
            indx2 = torch.tensor(indx2).cuda()
            source_features = features_test[indx1]  
            source_labels = label[indx1]
            target_features = features_test[indx2]



            cons_loss = (-softmax_out_old * torch.log(
                    softmax_out)).sum(1) - (softmax_out * torch.log(
                        softmax_out_old)).sum(1)
            total_loss += torch.mean(cons_loss)


            source_features1,y_a,y_b,lam=mixup_data(source_features,source_labels,1.0,True)
            source_features1,y_a,y_b = map(Variable,(source_features1,y_a,y_b))

                              
            
            dis_loss1 = netD(source_features,True)
            dis_loss2 = netD(target_features,False)
            dis_loss3 = netD(source_features1,True)
            dis_loss = dis_loss1+dis_loss2+dis_loss3
            total_loss+=0*dis_loss

            optimizer.zero_grad()
            optimizer_d.zero_grad()
            total_loss.backward()
            optimizer.step()
            optimizer_d.step()

        if True:
            total_loss = 0
            features_test = netB(netF(inputs_test))
            outputs_test_old = oldC(features_test)
            outputs_test = netC(features_test)

            softmax_out_old = nn.Softmax(dim=1)(outputs_test_old)
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            label = softmax_out_old.argmax(axis=1)
            
            msoftmax = softmax_out_old.mean(dim=0)
            div_loss = torch.sum(msoftmax * torch.log(msoftmax+1e-5))
            total_loss += div_loss

            msoftmax = softmax_out.mean(dim=0)
            div_loss = torch.sum(msoftmax * torch.log(msoftmax+1e-5))
            total_loss += div_loss
                  
            entropy_old = Entropy(softmax_out_old)
            indx = [i for i in range(batch_size)]
            indx1 = []
            indx2 = []
            for i in range(args.class_num):
                entropy_sub = entropy_old[label == i]
                if len(entropy_sub) == 0:
                    continue
                indx_sub = entropy_sub.topk(1, largest=False)[-1]
                for j in range(len(indx_sub)):
                    indx1.append(indx_sub[j].cpu().detach().tolist())
            for i in range(batch_size):
                if indx[i] in indx1:
                    continue
                else:
                    indx2.append(indx[i])
            indx1 = torch.tensor(indx1).cuda()
            indx2 = torch.tensor(indx2).cuda()
            source_features = features_test[indx1]   
            source_labels = label[indx1]
            target_features = features_test[indx2]
            
            softmax_source1 = netC(source_features)
            softmax_target = netC(target_features)
            criterion = nn.CrossEntropyLoss()

            ce_loss1 = criterion(softmax_source1,source_labels)



            #mixup
            source_features,y_a,y_b,lam=mixup_data(source_features,source_labels,1.0,True)
            source_features,y_a,y_b = map(Variable,(source_features,y_a,y_b))
            

            softmax_source = netC(source_features)


            ce_loss2 = mixup_criterion(criterion,softmax_source,y_a,y_b,lam)

            pred = pse_label[tar_idx]
            ce_loss3 = criterion(softmax_target, pred[indx2])
            if iter_num < interval_iter:
                ce_loss3 = 0


            ce_loss = ce_loss1+ce_loss2+ce_loss3

            
            total_loss += ce_loss
            optimizer.zero_grad()
            optimizer_c.zero_grad()
            total_loss.backward()
            optimizer.step()
            optimizer_c.step()

        if iter_num % int(args.interval * len(dset_loaders["target"])) == 0:
            # change += 1
            netF.eval()
            netB.eval()
            netC.eval()
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC,
                                    args.dset == "visda17")
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, \
                args.max_epoch * len(dset_loaders["target"]), acc) + '\n' + str(acc_list)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            netF.train()
            netB.train()
            netC.train()
            
    return netF, netB, netC

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PS on VisDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='visda-2017', choices=['visda-2017'])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101', help="resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2019, help="random seed")
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--threshold', type=int, default=0)


    parser.add_argument('--max_epoch', type=int, default=60, help="max iterations")
    parser.add_argument('--interval', type=float, default=0.1, help="max iterations")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='ps')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--zz', type=str, default='val', choices=['5', '10', '15', '20', '25', '30', 'val'])
    parser.add_argument('--savename', type=str, default='ps')
    args = parser.parse_args()

    args.interval = args.max_epoch / 30

    names = ['train', 'validation']
    args.class_num = 12


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    #torch.backends.cudnn.deterministic = True

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = args.t_dset_path

    current_folder = "./ckps/"
    args.output_dir_src = osp.join(current_folder, args.da, args.output, args.dset, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.output_dir = osp.join(current_folder, args.da, args.output, args.dset, names[args.s][0].upper() + names[args.t][0].upper())
    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(osp.join(args.output_dir, 'log_' + str(args.zz) + '_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target_ps(args, 'val')

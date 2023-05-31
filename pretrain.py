import argparse
import os

import torch
from torch.utils.data import DataLoader

from tool import label_abs2relative,setup_seed
from dataset.miniimagenet import MiniImageNet_Specific
from dataset.cifar100 import Cifar100_Specific
from dataset.cub import CUB_Specific
from dataset.flower import flower_Specific
from network import Conv4, ResNet18, ResNet10

parser = argparse.ArgumentParser(description='pretrain')
parser.add_argument('--dataset', type=str, default='cifar100', help='cifar100/miniimagenet/cub/flower')
parser.add_argument('--num_class_per_task', type=int, default=5, help='num_class_per_task')
parser.add_argument('--gpu', type=int, default=0, help="gpu")
parser.add_argument('--pre_backbone', type=str, default='conv4', help='conv4/resnet10/resnet18')
parser.add_argument('--multigpu', type=str, default='1', help='seen gpu')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.multigpu
setup_seed(2022)
device = torch.device('cuda:{}'.format(args.gpu))
if args.dataset=='cifar100' or args.dataset=='miniimagenet':
    args.num_node=13
    args.numclass=64
elif args.dataset=='cub':
    args.num_node=20
    args.numclass = 100
elif args.dataset=='flower':
    args.dataset=15
    args.numclass = 71
for node_id in range(args.num_node):
    node_classes = list(range(node_id * args.num_class_per_task, min(args.numclass, node_id * args.num_class_per_task + args.num_class_per_task)))
    if args.dataset=='cifar100':
        train_dataset = Cifar100_Specific(setname='meta_train', specific=node_classes, mode='train')
        test_dataset = Cifar100_Specific(setname='meta_train', specific=node_classes, mode='test')
    elif args.dataset=='miniimagenet':
        train_dataset = MiniImageNet_Specific(setname='meta_train', specific=node_classes, mode='train')
        test_dataset = MiniImageNet_Specific(setname='meta_train', specific=node_classes, mode='test')
    elif args.dataset=='cub':
        train_dataset = CUB_Specific(setname='meta_train', specific=node_classes, mode='train')
        test_dataset = CUB_Specific(setname='meta_train', specific=node_classes, mode='test')
    elif args.dataset=='flower':
        train_dataset = flower_Specific(setname='meta_train', specific=node_classes, mode='train')
        test_dataset = flower_Specific(setname='meta_train', specific=node_classes, mode='test')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
    #change the num_epoch and learning_rate when necessary
    num_epoch = 60
    learning_rate = 0.01
    if args.pre_backbone == 'conv4':
        if args.dataset == 'cifar100':
            teacher = Conv4(flatten=True, out_dim=args.num_class_per_task, img_size=32).cuda()
        else:
            teacher = Conv4(flatten=True, out_dim=args.num_class_per_task, img_size=84).cuda()
        optimizer = torch.optim.Adam(params=teacher.parameters(), lr=learning_rate)
    elif args.backbone == 'resnet18':
        teacher = ResNet18(flatten=True, out_dim=args.num_class_per_task).cuda()
        optimizer = torch.optim.SGD(params=teacher.parameters(), lr=learning_rate, momentum=.9, weight_decay=5e-4)
    elif args.backbone == 'resnet10':
        teacher = ResNet10(flatten=True, out_dim=args.num_class_per_task).cuda()
        optimizer = torch.optim.SGD(params=teacher.parameters(), lr=learning_rate, momentum=.9, weight_decay=5e-4)
    else:
        raise NotImplementedError

    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 40, 50], gamma=0.2)



    # train
    best_pre_model = None
    best_acc = None
    best_epoch=None
    not_increase = 0
    for epoch in range(num_epoch):
        # train
        teacher.train()
        for batch_count, batch in enumerate(train_loader):
            optimizer.zero_grad()
            image, abs_label = batch[0].cuda(), batch[1].cuda()
            relative_label = label_abs2relative(specific=node_classes, label_abs=abs_label).cuda()
            logits = teacher(image)
            criteria = torch.nn.CrossEntropyLoss()
            loss = criteria(logits, relative_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 50)
            optimizer.step()
        lr_schedule.step()
        correct, total = 0, 0
        teacher.eval()
        for batch_count, batch in enumerate(test_loader):
            image, abs_label = batch[0].cuda(), batch[1].cuda()
            relative_label = label_abs2relative(specific=node_classes, label_abs=abs_label).cuda()
            logits = teacher(image)
            prediction = torch.max(logits, 1)[1]
            correct = correct + (prediction.cpu() == relative_label.cpu()).sum()
            total = total + len(relative_label)
        test_acc = 100 * correct / total
        if best_acc == None or best_acc < test_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_pre_model = teacher.state_dict()
            not_increase = 0
        else:
            not_increase = not_increase + 1
            if not_increase == 60:
                print('early stop at:', best_epoch)
                break
        print('epoch{}acc:'.format(epoch), test_acc, 'best{}acc:'.format(best_epoch), best_acc)
    pretrain_path = './pretrained/{}/{}/{}/{}way/model_try'.format(args.dataset, args.pre_backbone, 'meta_train',args.num_class_per_task)
    os.makedirs(pretrain_path,exist_ok=True)
    torch.save(best_pre_model,os.path.join(pretrain_path,'model_{}.pth'.format(node_id)))
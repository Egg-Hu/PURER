import collections
import copy
import os
import random
import time

from torch.utils.data import DataLoader

import network
from dataset.cifar100 import Cifar100
from dataset.miniimagenet import MiniImageNet
from dataset.samplers import CategoriesSampler,CategoriesSampler_star
from dataset.cub import CUB
from dataset.flower import flower
import numpy as np
from network import *
def get_dataloader(args,noTransform_test=False):
    if args.dataset == 'cifar100':
        trainset = Cifar100(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size=trainset.img_size
        if args.maml_star==False:
            train_sampler = CategoriesSampler(trainset.label,
                                              args.episode_train,
                                              args.way_train,
                                              args.num_sup_train + args.num_qur_train)
        elif args.maml_star==True:
            train_sampler = CategoriesSampler_star(trainset.label,
                                              args.episode_train,
                                              args.way_train,
                                              args.num_sup_train + args.num_qur_train)
        train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        valset=Cifar100(setname='meta_val', augment=False)
        val_sampler = CategoriesSampler(valset.label,
                                          args.episode_test,
                                          args.way_test,
                                          args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                  num_workers=8,
                                  batch_sampler=val_sampler,
                                  pin_memory=True)
        testset = Cifar100(setname='meta_test', augment=False,noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                num_workers=8,
                                batch_sampler=test_sampler,
                                pin_memory=True)
    elif args.dataset == 'miniimagenet':
        trainset = MiniImageNet(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        if args.maml_star == False:
            train_sampler = CategoriesSampler(trainset.label,
                                              args.episode_train,
                                              args.way_train,
                                              args.num_sup_train + args.num_qur_train)
        elif args.maml_star == True:
            train_sampler = CategoriesSampler_star(trainset.label,
                                                   args.episode_train,
                                                   args.way_train,
                                                   args.num_sup_train + args.num_qur_train)
        train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        valset = MiniImageNet(setname='meta_val', augment=False)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = MiniImageNet(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
    elif args.dataset=='cub':
        trainset = CUB(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        assert args.num_classes==100
        args.img_size = trainset.img_size
        train_sampler = CategoriesSampler(trainset.label,
                                              args.episode_train,
                                              args.way_train,
                                              args.num_sup_train + args.num_qur_train)
        train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        valset = CUB(setname='meta_val', augment=False)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = CUB(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
    elif args.dataset=='flower':
        trainset = flower(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        assert args.num_classes==71
        args.img_size = trainset.img_size
        train_sampler = CategoriesSampler(trainset.label,
                                              args.episode_train,
                                              args.way_train,
                                              args.num_sup_train + args.num_qur_train)
        train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        valset = flower(setname='meta_val', augment=False)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = flower(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
    else:
        ValueError('not implemented!')
    return train_loader, val_loader, test_loader
def set_maml(flag):
    network.ConvBlock.maml = flag
    network.SimpleBlock.maml = flag
    network.BottleneckBlock.maml = flag
    network.ResNet.maml = flag
    network.ConvNet.maml = flag
def get_model(args,mode='train',set_maml_value=True,arbitrary_input=False):
    set_maml(set_maml_value)
    if mode=='train':
        way=args.way_train
    else:
        way = args.way_test
    if args.backbone == 'conv4':
        if args.dataset=='omniglot':
            channel=1
        else:
            channel=3
        model_maml = Conv4(flatten=True, out_dim=way, img_size=args.img_size,arbitrary_input=arbitrary_input,channel=channel)
    elif args.backbone == 'resnet34':
        model_maml = ResNet34(flatten=True, out_dim=way)
    elif args.backbone == 'resnet18':
        model_maml = ResNet18(flatten=True, out_dim=way)
    elif args.backbone == 'resnet50':
        model_maml = ResNet50(flatten=True, out_dim=way)
    elif args.backbone=='resnet10':
        model_maml = ResNet10(flatten=True, out_dim=way)
    else:
        ValueError('not implemented!')
    return model_maml
def get_premodel(args,mode='train',node_id=0):
    set_maml(False)
    if mode=='train':
        way=args.way_train
    else:
        way = args.way_test
    if args.pre_backbone == 'conv4':
        model_maml = Conv4(flatten=True, out_dim=way, img_size=args.img_size)
    elif args.pre_backbone == 'resnet34':
        model_maml = ResNet34(flatten=True, out_dim=way)
    elif args.pre_backbone == 'resnet18':
        model_maml = ResNet18(flatten=True, out_dim=way)
    elif args.pre_backbone == 'resnet50':
        model_maml = ResNet50(flatten=True, out_dim=way)
    elif args.pre_backbone=='resnet10':
        model_maml = ResNet10(flatten=True, out_dim=way)
    elif args.pre_backbone=='mix':
        pre_backbone_list = ['conv4', 'resnet10', 'resnet18']
        pre_backbone = pre_backbone_list[node_id % len(pre_backbone_list)]
        if pre_backbone == 'conv4':
            model_maml = Conv4(flatten=True, out_dim=way, img_size=args.img_size)
        elif pre_backbone == 'resnet34':
            model_maml = ResNet34(flatten=True, out_dim=way)
        elif pre_backbone == 'resnet18':
            model_maml = ResNet18(flatten=True, out_dim=way)
        elif pre_backbone == 'resnet50':
            model_maml = ResNet50(flatten=True, out_dim=way)
        elif pre_backbone == 'resnet10':
            model_maml = ResNet10(flatten=True, out_dim=way)
    else:
        ValueError('not implemented!')
    return model_maml
def set_zero(model_maml,args):
    for p1, p2 in enumerate(model_maml.parameters()):
        if args.backbone == 'conv4':
            if p1 == 16 or p1 == 17:
                p2.data = torch.zeros_like(p2.data)
        else:
            ValueError('not implement')
def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm
def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


NORMALIZE_DICT = {
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar10': dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'cifar100': dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    'miniimagenet': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'cub': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'flower': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'tinyimagenet': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    'cub200': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'stanford_dogs': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'stanford_cars': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365_32x32': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365_64x64': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'svhn': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'tiny_imagenet': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'imagenet_32x32': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # for semantic segmentation
    'camvid': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'nyuv2': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor
class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)
def get_transform(augment=False,dataset='cifar100'):
    if dataset=='cifar100':
        if augment:
            transforms_list = [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        else:
            transforms_list = [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        transform = transforms.Compose(
            transforms_list
        )
    elif dataset=='miniimagenet':
        transforms_list = [
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        transform = transforms.Compose(
            transforms_list
        )
    elif dataset=='cub':
        transforms_list = [
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        transform = transforms.Compose(
            transforms_list
        )
    elif dataset=='flower':
        transforms_list = [
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        transform = transforms.Compose(
            transforms_list
        )
    else:
        ValueError('not implement!')
    return transform

def label_abs2relative(specific,label_abs):
    trans=dict()
    for relative,abs in enumerate(specific):
        trans[abs]=relative
    label_relative=[]
    for abs in label_abs:
        label_relative.append(trans[abs.item()])
    return torch.LongTensor(label_relative)
def data2supportquery(args,mode,data):
    if mode=='train':
        way=args.way_train
        num_sup=args.num_sup_train
        num_qur=args.num_qur_train
    else:
        way = args.way_test
        num_sup = args.num_sup_test
        num_qur = args.num_qur_test
    label = torch.arange(way, dtype=torch.int16).repeat(num_qur+num_sup)
    label = label.type(torch.LongTensor)
    label = label.cuda()
    support=data[:way*num_sup]
    support_label=label[:way*num_sup]
    query=data[way*num_sup:]
    query_label=label[way*num_sup:]
    return support,support_label,query,query_label


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images(imgs, col=col).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        # output_filename = output.strip('.png')
        output_filename = output[:-4]
        for idx, img in enumerate(imgs):
            img = Image.fromarray(img.transpose(1, 2, 0).squeeze())
            img.save(output_filename + '-%d.png' % (idx))


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.m = mean
        self.v = var
        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()
def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)

def get_images_loss(net, bs=5, epochs=1000, idx=0, var_scale=0.001,
               net_student=None, competitive_scale=0.01,
               use_amp=False, bn_reg_scale = 10,l2_coeff=0.0,target=None,lr=0.25,first_bn_multiplier=10,pre_inputs=None,getLoss=False):
    #cudnn.benchmark = True
    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    # preventing backpropagation through student for Adaptive DeepInversion
    data_type = torch.half if use_amp else torch.float
    if pre_inputs==None:
        inputs = torch.randn((bs, 3, 32,32), requires_grad=True, device='cuda', dtype=data_type)
    else:
        inputs=pre_inputs
    net = copy.deepcopy(net)
    net.eval()

    if net_student!=None:
        net_student=copy.deepcopy(net_student)
        net_student.eval()
    if getLoss==False:
        optimizer = torch.optim.Adam([inputs], lr=lr)
    best_cost = 1e6

    # set up criteria for optimization
    criterion = nn.CrossEntropyLoss()
    if getLoss == False:
        optimizer.state = collections.defaultdict(dict)  # Reset state of optimizer
    lr_scheduler = lr_cosine_policy(lr, 100, epochs)

    # target outputs to generate
    targets=torch.LongTensor(target * (int(bs / len(target)))).to('cuda')

    ## Create hooks for feature statistics catching
    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # setting up the range for jitter
    lim_0, lim_1 = 2, 2
    count=0
    for epoch in range(epochs):
        # apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs, shifts=(off1,off2), dims=(2,3))
        if getLoss == False:
            lr_scheduler(optimizer, epoch, epoch)
        # foward with jit images
        if getLoss == False:
            optimizer.zero_grad()
        net.zero_grad()
        outputs = net(inputs_jit)
        loss = criterion(outputs, targets)
        # if getLoss==True:
        #     return loss
        loss_target = loss.item()

        # competition loss, Adaptive DeepInvesrion
        if competitive_scale != 0.0:
            net_student.zero_grad()
            outputs_student = net_student(inputs_jit)
            T = 3.0

            if 1:
                # jensen shanon divergence:
                # another way to force KL between negative probabilities
                P = F.softmax(outputs_student / T, dim=1)
                Q = F.softmax(outputs / T, dim=1)
                M = 0.5 * (P + Q)

                P = torch.clamp(P, 0.01, 0.99)
                Q = torch.clamp(Q, 0.01, 0.99)
                M = torch.clamp(M, 0.01, 0.99)
                eps = 0.0
                loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                # JS criteria - 0 means full correlation, 1 - means completely different
                loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

                loss = loss + competitive_scale * loss_verifier_cig

        # apply total variation regularization
        diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
        diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
        diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
        diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss = loss + var_scale*loss_var

        # R_feature loss
        rescale = [first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
        loss_distr = sum([mod.r_feature * rescale[idxxx] for (idxxx, mod) in enumerate(loss_r_feature_layers)])
        loss = loss + bn_reg_scale*loss_distr # best for noise before BN

        # l2 loss
        if 1:
            loss = loss + l2_coeff * torch.norm(inputs_jit, 2)

        if getLoss==True:
            return loss
        else:
            pass
        if best_cost > loss.item():
            best_cost = loss.item()
            best_inputs = copy.deepcopy(inputs.data)
            #print(count)
            best_epoch=epoch
            count=0
        else:
            count=count+1
        # backward pass
        loss.backward()

        optimizer.step()
    outputs=net(best_inputs)
    _, predicted_teach = outputs.max(1)
    if net_student != None:
        outputs_student=net_student(best_inputs)
        _, predicted_std = outputs_student.max(1)
    if net_student!=None:
        net_student.train()

    return best_inputs,targets

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def generate_random_task_data_fromDI(allData,allData_label,task_classes,num_sup,num_qur):
    supports, s_labels = [], []
    querys, q_labels = [], []
    for i in range(len(task_classes)):
        data_c = allData[allData_label == task_classes[i]]
        select = random.sample(range(data_c.shape[0]), num_sup + num_qur)
        data_c_select = data_c[select]
        support = data_c_select[:num_sup]
        query = data_c_select[num_sup:]
        supports.append(support)
        s_labels.append(torch.LongTensor(np.full((support.shape[0]), i)))
        querys.append(query)
        q_labels.append(torch.LongTensor(np.full((query.shape[0]), i)))
    con_support, con_s_label = cat(supports, s_labels)
    con_query, con_q_label = cat(querys, q_labels)
    return con_support, con_s_label,con_query, con_q_label
def cat(datas, labels):
    con_data = datas[0]
    con_label = labels[0]
    for i in range(1, len(datas)):
        con_data = torch.cat([con_data, datas[i]], dim=0)
        con_label = torch.cat([con_label,labels[i]], dim=0)
    return con_data, con_label
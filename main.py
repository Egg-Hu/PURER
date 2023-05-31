import argparse
from datetime import datetime
import math
import random
import torch
import os
from torch import nn
from tensorboardX import SummaryWriter
from tool import get_dataloader, get_model, set_zero, compute_confidence_interval, \
    get_premodel, data2supportquery, NORMALIZE_DICT,Normalizer,save_image_batch,setup_seed,FedAvg,Timer,get_images_loss,generate_random_task_data_fromDI
from methods.maml import Maml
parser = argparse.ArgumentParser(description='purer')
#basic
parser.add_argument('--multigpu', type=str, default='0', help='seen gpu')
parser.add_argument('--gpu', type=int, default=0, help="gpu")
parser.add_argument('--dataset', type=str, default='cifar100', help='cifar100/miniimagenet/cub/flower')
#maml
parser.add_argument('--way_train', type=int, default=5, help='way')
parser.add_argument('--num_sup_train', type=int, default=5)
parser.add_argument('--num_qur_train', type=int, default=15)
parser.add_argument('--way_test', type=int, default=5, help='way')
parser.add_argument('--num_sup_test', type=int, default=5)
parser.add_argument('--num_qur_test', type=int, default=15)
parser.add_argument('--backbone', type=str, default='conv4', help='conv4/resnet34/resnet18')
parser.add_argument('--episode_train', type=int, default=60000)
parser.add_argument('--start_id', type=int, default=1)
parser.add_argument('--episode_test', type=int, default=600)
parser.add_argument('--inner_update_num', type=int, default=5)
parser.add_argument('--test_inner_update_num', type=int, default=10)
parser.add_argument('--inner_lr', type=float, default=0.01)
parser.add_argument('--outer_lr', type=float, default=0.001)
parser.add_argument('--approx', action='store_true',default=False)
parser.add_argument('--episode_batch',type=int, default=4)
parser.add_argument('--efil', action='store_true',default=False)
parser.add_argument('--zero_ini', action='store_true',default=False)
parser.add_argument('--zero_train', action='store_true',default=False)
parser.add_argument('--zero_test', action='store_true',default=False)
parser.add_argument('--zeroExactLoss', action='store_true',default=False)
parser.add_argument('--zeroExactLossNoInner', action='store_true',default=False)
parser.add_argument('--val_interval',type=int, default=2000)
parser.add_argument('--save_interval',type=int, default=2000)
parser.add_argument('--maml_star', action='store_true',default=False)
#dfmeta
parser.add_argument('--pre_backbone', type=str, default='conv4', help='conv4/resnet34/resnet18')
parser.add_argument('--average', action='store_true',default=False)
parser.add_argument('--synthesis_per_class',type=int, default=30)
parser.add_argument('--adv', action='store_true',default=False)
parser.add_argument('--adv_scale', type=float, default=1.0)
parser.add_argument('--adv_start', type=int, default=1)
parser.add_argument('--all_class',type=int,default=64)


parser.add_argument('--attack', action='store_true',default=False)



args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.multigpu
setup_seed(2022)


device=torch.device('cuda:{}'.format(args.gpu))
_,_,test_loader=get_dataloader(args)
args.all_class=args.num_classes
args.num_node_meta_train=math.ceil(args.all_class/float(args.way_train))



if args.dataset=='cifar100':
    allData = torch.randn([args.synthesis_per_class*args.all_class,3,32,32], requires_grad=True, device='cuda', dtype=torch.float)
elif args.dataset=='miniimagenet':
    allData = torch.randn([args.synthesis_per_class*args.all_class,3,84,84], requires_grad=True, device='cuda', dtype=torch.float)
elif args.dataset=='cub':
    allData = torch.randn([args.synthesis_per_class*args.all_class,3,84,84], requires_grad=True, device='cuda', dtype=torch.float)
elif args.dataset == 'flower':
    allData = torch.randn([args.synthesis_per_class * args.all_class, 3, 84, 84], requires_grad=True, device='cuda',dtype=torch.float)
else:
    ValueError('not implemented!')
allData_label = []
for i in range(args.num_node_meta_train):
    node_classes = list(range(i * args.way_train, min(args.all_class,i * args.way_train + args.way_train)))
    load_data_di_label = torch.LongTensor([l for l in node_classes] * (args.synthesis_per_class))
    allData_label.append(load_data_di_label)
allData_label = torch.cat(allData_label, 0)

model_maml=get_model(args,'train',arbitrary_input=False)
if args.average==True and args.pre_backbone!='mix':
    pretrained_model_list = []
    for pretrain_id in range(args.num_node_meta_train):
        pretrained_model_list.append(torch.load('./pretrained/{}/{}/{}/{}way/model_try/model_{}.pth'.format(args.dataset, args.pre_backbone,'meta_train', args.way_train, pretrain_id)))
    model_maml_dic = FedAvg(pretrained_model_list)
    model_maml.load_state_dict(model_maml_dic)
    del model_maml_dic
model_maml.cuda(device)
if args.zero_ini:
    set_zero(model_maml, args)
optimizer = torch.optim.Adam(params=model_maml.parameters(), lr=args.outer_lr)
optimizer_di = torch.optim.Adam([allData], lr=0.25)
criteria = nn.CrossEntropyLoss()
maml=Maml(args)
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
feature='{}_{}w_{}s_{}q_{}_{}_{}outer_{}inner_{}EB_{}B_{}PB'.format(args.dataset,args.way_train, args.num_sup_train,args.num_qur_train, args.inner_update_num,
                                                                       args.test_inner_update_num,args.outer_lr,args.inner_lr,args.episode_batch,args.backbone,args.pre_backbone)
if args.average and args.pre_backbone!='mix':
    feature = feature + '_Average'
if args.maml_star:
    feature = feature + '_Star'
if args.approx:
    feature = feature + '_1Order'
if args.efil:
    feature = feature + '_EFIL'
if args.zero_ini:
    feature = feature + '_Zero_ini'
if args.zero_train:
    feature = feature + '_Zero_train'
if args.zero_test:
    feature = feature + '_Zero_test'
if args.zeroExactLoss:
    feature = feature + '_ZeroExactLoss'
if args.zeroExactLossNoInner:
    feature = feature + '_ZeroExactLossNoInner'
if args.adv:
    feature = feature + '_Adv{}StartIT{}'.format(args.adv_scale,args.adv_start)
if args.start_id!=1:
    feature = feature + '_Startfrom{}'.format(args.start_id)
writer_path = './log/' + TIMESTAMP+'/'+feature
#train

max_acc_val=0
maxAcc = None
generate_update=True
adv_update =False
timer = Timer()
teacher=get_premodel(args,'train').cuda()
min_loss=10000
target=None
with SummaryWriter(writer_path) as writer:
    loss_batch,acc_batch=[],[]
    for task_id in range(args.start_id,args.episode_train+1):
        #generate
        if generate_update==True:
            loss_data = 0
            loss2=0
            for node_id in range(args.num_node_meta_train):
                node_classes = list(range(node_id * args.way_train, min(args.all_class,node_id * args.way_train + args.way_train)))
                #original
                teacher = get_premodel(args, 'train',node_id=node_id)
                teacher.cuda(device)
                teacher.load_state_dict(torch.load('./pretrained/{}/{}/{}/{}way/model_try/model_{}.pth'.format(args.dataset, args.pre_backbone,'meta_train', args.way_train, node_id)))
                if args.dataset == 'cifar100':
                    loss_data = loss_data + get_images_loss(net=teacher,
                                                             bs=args.synthesis_per_class * len(node_classes),
                                                             epochs=20000, idx=0,
                                                             var_scale=0.0001, net_student=None,
                                                             competitive_scale=0.0,
                                                             use_amp=False, bn_reg_scale=0.01,#0.01
                                                             l2_coeff=0.00001,
                                                             target=[l for l in range(len(node_classes))],
                                                             lr=0.25, first_bn_multiplier=1,
                                                             pre_inputs=allData[
                                                                        node_id * args.way_train * args.synthesis_per_class:min(
                                                                            (node_id + 1) * args.way_train * args.synthesis_per_class,
                                                                            allData.shape[0])],
                                                             getLoss=True)
                elif args.dataset == 'miniimagenet':
                    loss_data = loss_data + get_images_loss(net=teacher,
                                                             bs=args.synthesis_per_class * len(node_classes),
                                                             epochs=20000, idx=0,
                                                             var_scale=0.0001, net_student=None,
                                                             competitive_scale=0.0,
                                                             use_amp=False, bn_reg_scale=0.01, l2_coeff=0.00001,
                                                             target=[l for l in range(len(node_classes))],
                                                             lr=0.25, first_bn_multiplier=1,
                                                             pre_inputs=allData[
                                                                        node_id * args.way_train * args.synthesis_per_class:min(
                                                                            (node_id + 1) * args.way_train * args.synthesis_per_class,
                                                                            allData.shape[0])],
                                                             getLoss=True)
                elif args.dataset=='cub':
                    loss_data = loss_data + get_images_loss(net=teacher,
                                                             bs=args.synthesis_per_class * len(node_classes),
                                                             epochs=20000, idx=0,
                                                             var_scale=0.0001, net_student=None,
                                                             competitive_scale=0.0,
                                                             use_amp=False, bn_reg_scale=0.01, l2_coeff=0.00001,
                                                             target=[l for l in range(len(node_classes))],
                                                             lr=0.25, first_bn_multiplier=1,
                                                             pre_inputs=allData[
                                                                        node_id * args.way_train * args.synthesis_per_class:min(
                                                                            (node_id + 1) * args.way_train * args.synthesis_per_class,
                                                                            allData.shape[0])],
                                                             getLoss=True)
                elif args.dataset=='flower':
                    loss_data = loss_data + get_images_loss(net=teacher,
                                                             bs=args.synthesis_per_class * len(node_classes),
                                                             epochs=20000, idx=0,
                                                             var_scale=0.0001, net_student=None,
                                                             competitive_scale=0.0,
                                                             use_amp=False, bn_reg_scale=0.01, l2_coeff=0.00001,
                                                             target=[l for l in range(len(node_classes))],
                                                             lr=0.25, first_bn_multiplier=1,
                                                             pre_inputs=allData[
                                                                        node_id * args.way_train * args.synthesis_per_class:min(
                                                                            (node_id + 1) * args.way_train * args.synthesis_per_class,
                                                                            allData.shape[0])],
                                                             getLoss=True)
                else:
                    raise NotImplementedError
                loss2=loss2+loss_data
            if args.adv == True and adv_update == True:
                task_classes = random.sample(range(args.all_class), args.way_train)
                support_data, support_label, query_data, query_label = generate_random_task_data_fromDI(allData, allData_label, task_classes, args.num_sup_train, args.num_qur_train)
                support, support_label, query, query_label = support_data.cuda(device), support_label.cuda(device), query_data.cuda(device), query_label.cuda(device)
                loss_outer, _ = maml.run(model_maml, support, support_label, query, query_label, nn.CrossEntropyLoss(),device, 'train')
                loss2 = loss2 + -1 * args.adv_scale * loss_outer
                adv_update=False
            optimizer_di.zero_grad()
            loss2.backward()
            optimizer_di.step()
            generate_update=False
            #visual
            # normalizer = Normalizer(**NORMALIZE_DICT[args.dataset])
            # save_image_batch(normalizer(x=allData[400:460].clone(), reverse=True),
            #                  './view/view'.format((task_id) // args.episode_batch) + '.png', col=5)
        pool = range(args.all_class)
        task_classes = random.sample(pool, args.way_train)
        support_data, support_label, query_data, query_label = generate_random_task_data_fromDI(allData.detach(),allData_label,task_classes, args.num_sup_train,args.num_qur_train)
        support, support_label, query, query_label = support_data.cuda(device), support_label.cuda(device), query_data.cuda(device), query_label.cuda(device)
        loss_outer, acc = maml.run(model_maml, support, support_label, query, query_label, nn.CrossEntropyLoss(),device, 'train')
        loss_batch.append(loss_outer)
        acc_batch.append(acc)
        if task_id % args.episode_batch == 0:
            generate_update=True
            if task_id//args.episode_batch==args.adv_start:
                adv_update=True
            loss = torch.stack(loss_batch).sum(0)
            acc = torch.stack(acc_batch).mean()
            if task_id//args.episode_batch>args.adv_start:
                if adv_update==False:
                    if maxAcc == None or acc > maxAcc:
                        maxAcc = acc
                        e_count = 0
                    else:
                        e_count = e_count + 1
                        if e_count == 6:
                            print('meta_update in it', (task_id ) // args.episode_batch)
                            adv_update = True
                            e_count = 0
                            maxAcc = None
            loss_batch, acc_batch = [], []
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if args.zero_train:
            set_zero(model_maml, args)
        # val
        if task_id % args.val_interval == 0:
            acc_val = []
            for test_batch in test_loader:
                data, g_label = test_batch[0].cuda(device), test_batch[1].cuda(device)
                support, support_label_relative, query, query_label_relative = data2supportquery(args, 'test', data)
                _, acc = maml.run(model_maml=model_maml, support=support, support_label=support_label_relative,
                                  query=query, query_label=query_label_relative, criteria=criteria, device=device,
                                  mode='test')
                acc_val.append(acc)
                if args.zero_test:
                    set_zero(model_maml, args)
            acc_val, pm = compute_confidence_interval(acc_val)
            if acc_val > max_acc_val:
                max_acc_val = acc_val
                max_it = (task_id) // args.episode_batch
                max_pm = pm
            print((task_id) // args.episode_batch, 'test acc:', acc_val, '+-', pm)
            print(max_it, 'best test acc:', max_acc_val, '+-', max_pm)
            print('ETA:{}/{}'.format(
                timer.measure(),
                timer.measure((task_id) / (args.episode_train)))
            )
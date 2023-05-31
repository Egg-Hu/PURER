import copy
import torch
import torch.nn.functional as F



class Maml():
    def __init__(self,args):
        self.args=args
    def run(self,model_maml,support,support_label,query,query_label,criteria,device,mode='train'):
        if mode=='train':
            model_maml.train()
            way=self.args.way_train
            num_sup=self.args.num_sup_train
            num_qur=self.args.num_qur_train
            inner_update_num=self.args.inner_update_num
        else:
            model_maml.train()
            way = self.args.way_test
            num_sup = self.args.num_sup_test
            num_qur = self.args.num_qur_test
            inner_update_num = self.args.test_inner_update_num

        # inner
        fast_parameters = list(model_maml.parameters())
        for weight in model_maml.parameters():
            weight.fast = None
        model_maml.zero_grad()
        correct, total = 0, 0
        for inner_step in range(inner_update_num):
            pred = model_maml(support)
            loss_inner = criteria(pred, support_label)
            grad = torch.autograd.grad(loss_inner, fast_parameters, create_graph=True)
            if self.args.approx == True:
                grad = [g.detach() for g in grad]
            fast_parameters = []
            for k, weight in enumerate(model_maml.parameters()):
                if weight.fast is None:
                    weight.fast = weight - self.args.inner_lr * grad[k]
                else:
                    weight.fast = weight.fast - self.args.inner_lr * grad[k]
                fast_parameters.append(weight.fast)
        # outer
        score = model_maml(query)
        prediction = torch.max(score, 1)[1]
        correct = correct + (prediction.cpu() == query_label.cpu()).sum()
        total = total + len(query_label)
        loss_outer = criteria(score, query_label)
        acc=1.0*correct/total*100.0
        return loss_outer,acc

import numpy as np
import torch
import random, os
import torch.nn.functional as F

from torch.optim import Optimizer

"""
https://github.com/coranholmes/TEVAD/blob/main/utils.py

"""

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int64)  # len=33,存入要取的frame index
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)  # r[i]:r[i+1]这些feat求平均
        else:
            new_feat[i, :] = feat[r[i], :]  # 不足32帧补全
    return new_feat


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=0)[0])
        min_val = relu(torch.min(act_map, dim=0)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def modelsize(model, input, type_size=4):
    # check GPU utilisation
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))


def save_best_record(test_info, file_path, metrics):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write(metrics + ": " +str(test_info[metrics][-1]))
    fo.close()


def vid_name_to_path(vid_name, mode):  # TODO: change absolute paths! (only used by visual codes)
    root_dir = '/home/acsguser/Codes/SwinBERT/datasets/Crime/data/'
    types = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery",
             "Shooting", "Shoplifting", "Stealing", "Vandalism"]
    for t in types:
        if vid_name.startswith(t):
            path = root_dir + t + '/' + vid_name
            return path
    if vid_name.startswith('Normal'):
        if mode == 'train':
            path = root_dir + 'Training_Normal_Videos_Anomaly/' + vid_name
        else:
            path = root_dir + 'Testing_Normal_Videos_Anomaly/' + vid_name
        return path
    raise Exception("Unknown video type!!!")


def seed_everything(seed=4869):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_rgb_list_file(ds, is_test):
    if "ucf" in ds:
        ds_name = "Crime"
        if is_test:
            rgb_list_file = 'list/ucf-i3d-test.list'
        else:
            rgb_list_file = 'list/ucf-i3d.list'
    elif "shanghai" in ds:
        ds_name = "Shanghai"
        if is_test:
            rgb_list_file = 'list/shanghai-i3d-test-10crop.list'
        else:
            rgb_list_file = 'list/shanghai-i3d-train-10crop.list' 
    elif "ped2" in ds:
        ds_name = "UCSDped2"
        if is_test:
            rgb_list_file = 'list/ped2-i3d-test.list'
        else:
            rgb_list_file = 'list/ped2-i3d.list'
    elif "violence" in ds:
        ds_name = "Violence"
        if is_test:
            rgb_list_file = 'list/violence-i3d-test.list'
        else:
            rgb_list_file = 'list/violence-i3d.list'
    else:
        raise ValueError("dataset should be either ucf, shanghai, or violence")
    return ds_name, rgb_list_file

def get_gt(ds, gt_file):
    if gt_file is not None:
        gt = np.load(gt_file)
    else:
        if 'shanghai' in ds:
            gt = np.load('list/gt-sh2.npy')
        elif 'ucf' in ds:
            gt = np.load('list/gt-ucf.npy')
        elif 'ped2' in ds:
            gt = np.load('list/gt-ped2.npy')
        elif 'violence' in ds:
            gt = np.load('list/gt-violence.npy')
        else:
            raise Exception("Dataset undefined!!!")
    return gt

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

@torch.jit.script
def lambda_x(x: torch.Tensor):
    return 2 / (1 - torch.sum(x ** 2, dim=-1, keepdim=True))

@torch.jit.script
def mobius_add(x: torch.Tensor, y: torch.Tensor):
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)

    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2

    return num / denom.clamp_min(1e-15)

@torch.jit.script
def expm(p: torch.Tensor, u: torch.Tensor):
    return p + u
    # for exact exponential mapping
    #norm = torch.sqrt(torch.sum(u ** 2, dim=-1, keepdim=True))
    #return mobius_add(p, torch.tanh(0.5 * lambda_x(p) * norm) * u / norm.clamp_min(1e-15))

@torch.jit.script
def grad(p: torch.Tensor):
    p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
    return p.grad.data * ((1 - p_sqnorm) ** 2 / 4).expand_as(p.grad.data)

class RiemannianSGD(Optimizer):
    def __init__(self, params):
        super(RiemannianSGD, self).__init__(params, {})

    def step(self, lr=0.3):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = grad(p)
                d_p.mul_(-lr)

                p.data.copy_(expm(p.data, d_p))


def arccosh( x):
        """
        Element-wise arcosh operation.
        Parameters
        ---
        x : torch.Tensor[]
        Returns
        ---
        torch.Tensor[]
            arcosh result.
        """
        return torch.log(x + torch.sqrt(torch.pow(x, 2) - 1))


def lorentz_similarity( x: torch.Tensor, y: torch.Tensor, c) -> torch.Tensor:
        '''
        d = <x, y>   lorentz metric   计算洛伦兹度量
        '''
        eps = {torch.float32: 1e-6, torch.float64: 1e-8}  #设置容错值
        idx = np.concatenate((np.array([-1]), np.ones(x.shape[-1] - 1)))    #创建对角矩阵
        diag = torch.from_numpy(np.diag(idx).astype(np.float32)).to(x.device)
        temp = x @ diag  #计算x与对角矩阵的乘法
        xy_inner = -(temp @ y.transpose(-1, -2))   #计算洛伦兹内积
        xy_inner_ = F.threshold(xy_inner, 1, 1)  #使用阈值函数 
        sqrt_k = c**0.5
        dist = sqrt_k * arccosh(xy_inner_ / c)  #计算洛伦兹距离
        dist = torch.clamp(dist, min=eps[x.dtype], max=200)   #将距离限制在一个合理的范围
        return dist

# adjacency matrix for hyperbolic space
def hyperbolic_adj(x, thres, c, seq_len=None): #previously: def hyperbolic_adj(x, thres, c, seq_len):
    softmax = torch.nn.Softmax(dim=-1)
    x2 = lorentz_similarity(x,x,c)
    x2 = torch.exp(-x2) # map similarity to [0,1]
    output = torch.zeros_like(x2)
    if seq_len is None:
        for i in range(x.shape[0]):
            tmp = x2[i]
            adj2 = tmp
            adj2 = F.threshold(adj2, thres[i].item(), 0) #thres[i].item() previous.
            adj2 = softmax(adj2)
            output[i] = adj2
    else:
        for i in range(len(seq_len)): # previously: for i in range(len(seq_len)):
            tmp = x2[i, :seq_len[i], :seq_len[i]] # previously: tmp = x2[i, :seq_len[i], :seq_len[i]]
            adj2 = tmp
            adj2 = F.threshold(adj2, thres[i].item(), 0)   #thres[i].item() previous, 0.8
            adj2 = softmax(adj2)
            output[i, :seq_len[i], :seq_len[i]] = adj2 # previously: output[i, :seq_len[i], :seq_len[i]] = adj2
    return output
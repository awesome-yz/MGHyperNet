import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torchhyp.pmath as pm
from utils import *


def sparsity(arr, lamda2): 
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss


def smooth(arr, lamda1, batch_size, sequence_length=32):  
    """
    Corrected non-leaking temporal smoothness loss.
    Reshapes flat predictions back to distinct video profiles before processing shifts.
    """
    # 1. Reshape the flat array back into structural blocks: [Batch, Video Segments]
    # This completely isolates individual video streams from each other
    grid_scores = arr.view(batch_size, sequence_length)
    
    # 2. Extract pairs across the temporal sequence dim (t versus t+1) safely
    current_frames = grid_scores[:, :-1]
    next_frames = grid_scores[:, 1:]
    
    # 3. Penalize adjacent frames natively inside their own separate streams
    loss = torch.sum((next_frames - current_frames) ** 2)
    return lamda1 * loss


class ContrastiveLoss(torch.nn.Module):
    """
    Hyperbolic contrastive loss.
    Implementation based on MGFN (https://github.com/carolchenyx/MGFN./blob/main/train.py)
    """
    def __init__(self, margin=1.0, c=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.c = c
        
    def forward(self, out1, out2, label, d=None):
        if d == "hyperbolic":
            # Native geodesic metric projection on the Poincare Ball manifold
            distance = torch.tanh(pm.dist(out1, out2, c=self.c))  
        else:
            distance = F.pairwise_distance(out1, out2, keepdim=True)
            
        loss_contrastive = torch.mean((1 - label) * torch.pow(distance, 2) + 
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive


class Loss(torch.nn.Module):
    def __init__(self, alpha, beta, margin, normal_weight, abnormal_weight, c, mag_loss=True, attn_loss=True):
        super(Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.normal_weight = normal_weight
        self.abnormal_weight = abnormal_weight
        self.c = c
        self.mag_loss = mag_loss
        self.attn_loss = attn_loss
        
        if normal_weight == 1 and abnormal_weight == 1:
            self.criterion = torch.nn.BCELoss()
        else:
            self.criterion = None
        self.contrastiveloss = ContrastiveLoss(c=self.c)

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a, mag_n, mag_a):
        n = int(len(feat_n) / 2)
        s = int(len(mag_n) / 2)
        label = torch.cat((nlabel, alabel), 0).cuda()  

        score = torch.cat((score_normal, score_abnormal), 0).squeeze()

        if self.normal_weight == 1 and self.abnormal_weight == 1:
            loss_cls = self.criterion(score, label)
        else:
            weight = self.abnormal_weight * label + self.normal_weight * (1.0 - label)
            loss_cls = torch.nn.functional.binary_cross_entropy(score, label, weight=weight)
    
        # Hyperbolic semantic space calculations continue normally
        loss_con = self.contrastiveloss(pm.poincare_mean(feat_a, dim=1), pm.poincare_mean(feat_n, dim=1), 1, d="hyperbolic")
        loss_con_n = self.contrastiveloss(pm.poincare_mean(feat_n[:n], dim=1), pm.poincare_mean(feat_n[n:], dim=1), 0, d="hyperbolic")
        loss_con_a = self.contrastiveloss(pm.poincare_mean(feat_a[:n], dim=1), pm.poincare_mean(feat_a[n:], dim=1), 0, d="hyperbolic")
        loss_con += loss_con_n + loss_con_a

        loss_mag = self.contrastiveloss(pm.poincare_mean(mag_a, dim=0), pm.poincare_mean(mag_n, dim=0), 1, d="hyperbolic") 
        loss_mag_n = self.contrastiveloss(pm.poincare_mean(mag_n[:s], dim=0), pm.poincare_mean(mag_n[s:], dim=0), 0, d="hyperbolic")
        loss_mag_a = self.contrastiveloss(pm.poincare_mean(mag_a[:s], dim=0), pm.poincare_mean(mag_a[s:], dim=0), 0, d="hyperbolic")
        loss_mag += loss_mag_n + loss_mag_a

        loss_total = loss_cls
        if self.mag_loss:
            loss_total += (self.alpha * loss_mag)
        if self.attn_loss:
            loss_total += (self.beta * loss_con)
        return loss_total


def train(nloader, aloader, model, args, optimizer, device, step, train_loss=None):
    with torch.set_grad_enabled(True):
        model.train()
        batch_size = args.batch_size
        ninput, ntext, nlabel = next(nloader)  
        ainput, atext, alabel = next(aloader)  

        input = torch.cat((ninput, ainput), 0).to(device)  
        text = torch.cat((ntext, atext), 0).to(device)

        seq_len = torch.sum(torch.max(torch.abs(input.view(-1, input.shape[-2], input.shape[-1])), dim=2)[0] > 0, 1)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores, mag_abn, mag_nor, c_out, _ = model(input, text, step=step, seq_len=seq_len)  
        
        if args.train_c:
            args.c = c_out.detach().cpu().item() 
        print("Updated curvature c: ", args.c)

        scores = scores.view(batch_size * 32 * 2, -1).squeeze()
        abn_scores = scores[batch_size * 32:]  

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        loss_criterion = Loss(args.alpha, 
                              args.beta, 
                              args.margin, 
                              args.normal_weight, 
                              args.abnormal_weight, 
                              c_out.detach(), 
                              args.mag_loss, 
                              args.attn_loss) 
        loss_sparse = sparsity(abn_scores, 8e-3) 
     
        loss_smooth = smooth(abn_scores, 8e-4, batch_size=batch_size, sequence_length=32)  
       
        cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn, mag_nor, mag_abn) + loss_smooth + loss_sparse
        print(f'loss_cls: {cost.item()}, loss_smooth: {loss_smooth.item()}, loss_sparse: {loss_sparse.item()}')
        
        train_loss.append(cost.item())

        optimizer.zero_grad(set_to_none=True)  
        cost.backward(retain_graph=True) 
        
        print(f'max_grad: {max(p.grad.abs().max() for p in model.parameters() if p.grad is not None)}') 

        current_lr = optimizer.param_groups[0]['lr']

         
        dynamic_clip = max(1.0, 10.0 * (current_lr / args.lr))
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=dynamic_clip)
        
        optimizer.step() 

        return train_loss

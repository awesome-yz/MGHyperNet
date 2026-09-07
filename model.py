import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torchhyp.nn as hypnn
import torchhyp.pmath as pm
import math
import torch.nn.functional as F
import torchhyp.lorentz as lorentz
from layers.hyp_layers import HyperbolicGraphConvolution as hypgcn
from torchhyp.lorentz import Lorentz
from torchhyp.hyperboloid import Hyperboloid 
import utils
import numpy as np
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class LinearSelfAttention(nn.Module):
    def __init__(self, args, out_dim=512):
        super(LinearSelfAttention, self).__init__()
        self.in_dim = args.feature_size
        self.out_dim = out_dim
        self.scale = 1.0 / math.sqrt(self.out_dim)
        
        self.q_proj = nn.Linear(self.in_dim, self.out_dim)
        self.k_proj = nn.Linear(self.in_dim, self.out_dim)
        self.v_proj = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        qs = self.q_proj(x)
        ks = self.k_proj(x)
        vs = self.v_proj(x)

        attn_weight = torch.matmul(qs, ks.transpose(-2, -1)) * self.scale
        score = F.softmax(attn_weight, dim=-1)
        attn = torch.matmul(score, vs)
        return attn


class HypSelfAttenion(nn.Module):
    def __init__(self, args, out_dim=512, c=1.0):
        super(HypSelfAttenion,self).__init__()
        """
        Hyperbolic self-attention module.
        Args:
            args: arguments
            out_dim: output dimension
            c: curvature

        """
        self.in_dim = args.feature_size
        self.out_dim = out_dim
        self.scale = nn.Parameter(torch.tensor(math.sqrt(self.out_dim))) 
        self.bias = nn.Parameter(torch.zeros(()))
        self.ch_bias = nn.Parameter(torch.zeros(()))
        self.c = c
        self.linear = hypnn.HypLinear(self.in_dim, self.out_dim, c=self.c)

    
    def forward(self, x, c=1.0):
        qs = self.linear(x, c)
        ks = self.linear(x, c)
        vs = self.linear(x, c)

        ks_transpose = torch.transpose(ks, -2, -1)
        qs_ks = torch.einsum('nid,ndj ->nij', qs, ks_transpose)
        attn_weight = qs_ks/self.scale + self.bias
        score = F.softmax(attn_weight, dim=-1)
        attn = torch.einsum('nij,njd->nid', score, vs)
        return attn

class Model(nn.Module):
    """
    Main model class. 
    Implementation based on TEVAD (https://github.com/coranholmes/TEVAD/blob/main/model.py)
    """
    def __init__(self, args):
        super(Model, self).__init__()

        self.batch_size = args.batch_size
        self.num_segments = 32
        self.k_abn = self.num_segments // 10  # top k for abnormal snippets
        self.k_nor = self.num_segments // 10  # top k for normal snippets
        self.ncrops = args.ncrops
        self.train_c = args.train_c
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=False)
        self.use_attn = args.use_attn
        self.text = args.text
        self.multimodal = args.multimodal
        self.dropout = nn.Dropout(args.dropout) # default 0.7

        # hyplinear input dimension:
        if self.multimodal:
            self.concat_feat_size = args.gcn_layers_vfeats[-1] + args.gcn_layers_tfeats[-1]
        elif self.text:
            self.concat_feat_size = args.gcn_layers_tfeats[-1]
        else:
            self.concat_feat_size = args.gcn_layers_vfeats[-1]
        if not self.multimodal and self.text:
            self.linear_feat_size = args.gcn_layers_tfeats[-1]
        else:
            self.linear_feat_size = args.linear_feat_size
        self.gcn_layers_vfeats = args.gcn_layers_vfeats
        self.gcn_layers_tfeats = args.gcn_layers_tfeats
        
        # setting hyperbolic curvature
        if self.train_c:
            self.c = nn.Parameter(torch.tensor(args.c).log(), requires_grad=self.train_c)
            print(f"Training curvature initial c: {self.c.exp().item()}")
            self._curv_minmax = {
            "max": -1e-5,  # exp(-0.00001) = 0.99999 -> Safe from division by zero!
            "min": -10.0  # exp(-10.0) = 0.000045 -> Soft lower floor boundary
            }
        else:
            self.c = torch.tensor(args.c).log()

        # visual and text embeddings to hyperbolic space
        self.hyp_liner1 =  hypnn.HypLinear(self.concat_feat_size, self.linear_feat_size, bias=False, c=self.c.exp())

        self.hyp_gcn_layers_vis = nn.ModuleList()
        for i in range(len(args.gcn_layers_vfeats)-1):
            self.hyp_gcn_layers_vis.append(hypgcn(manifold=args.manifold,
                                                  in_features=args.gcn_layers_vfeats[i] + 1,
                                                  out_features=args.gcn_layers_vfeats[i + 1] + 1,
                                                  c_in=self.c.exp(),
                                                  c_out=self.c.exp(),
                                                  dropout=args.gcn_dropout,
                                                  act=F.relu,
                                                  use_bias=True,
                                                  use_att=args.use_gcn_attn,
                                                  local_agg=args.local_agg))
        

        self.hyp_gcn_layers_txt = nn.ModuleList()
        for i in range(len(args.gcn_layers_tfeats) - 1):
            self.hyp_gcn_layers_txt.append(hypgcn(manifold=args.manifold,
                                                  in_features=args.gcn_layers_tfeats[i] + 1,
                                                  out_features=args.gcn_layers_tfeats[i + 1] + 1,
                                                  c_in=self.c.exp(),
                                                  c_out=self.c.exp(),
                                                  dropout=args.gcn_dropout,
                                                  act=F.relu,
                                                  use_bias=True,
                                                  use_att=args.use_gcn_attn,
                                                  local_agg=args.local_agg))
        
        
        self.lorentz = Lorentz()
        self.hyperboloid = Hyperboloid()
        
        self.res_vis = hypnn.ToPoincare(train_c=args.train_c, train_x=args.train_x, ball_dim=args.feature_size, c=self.c.exp())
        self.to_euclidean = hypnn.FromPoincare(train_c=args.train_c, train_x=args.train_x, ball_dim=args.feature_size, c=self.c.exp())
        
        # concat  hyperbolic visual and text features (poincare)
        self.concat = hypnn.ConcatPoincareLayer(d1=args.gcn_layers_vfeats[-1], d2=args.gcn_layers_tfeats[-1], d_out=self.concat_feat_size, c=self.c.exp())

    
    def clip_to_ball(self, x, max_norm=1 - 1e-5):
        '''
        Clip the input tensor to the Poincare ball of radius max_norm.
        Args:
            x: input tensor
            max_norm: maximum norm of the Poincare ball
        Returns:
            x: clipped tensor'''
        
        norm = torch.norm(x, dim=-1, keepdim=True)
        scaling = torch.clamp(norm / max_norm, min=1.0)
        return x / scaling
    
    def hyp_encoder(self,encoder, x, c): # mapping to hyperbolic space (lorentz model)
        o = torch.zeros_like(x)
        x = torch.cat([o[:,:,0:1], x], dim=-1)  # add the first dimension
        x = encoder.expmap0(x,c)
        return x

    def calculate_LSHA(self,E_x, k,beta=0.8,gamma=1.2):
        he = 1/(E_x+1)
        LSHA = torch.sigmoid(beta*k-gamma+he)
        return LSHA
    
    def sigmoid1(self,x):
        return 1 / (1 + np.exp(-x))

    def calculate_hyperbolic_dirichlet_energy(self,feature, c):
        b,n,d = feature.shape
        d_lorentz = utils.lorentz_similarity(feature,feature,c)
        # Sum across the last two dimensions
        d_summed = torch.sum(d_lorentz[:,:,], axis=2)
        d_summed = torch.sum(d_summed[:,],axis = 1)

        average = d_summed/(2*n)
        return average

    def forward(self, inputs, text, step=None, seq_len=None):
        if self.train_c:
            self.c.data = torch.clamp(self.c, **self._curv_minmax)  
    
        k_abn = self.k_abn # number of top k snippets for abnormal bag
        k_nor = self.k_nor # number of top k snippets for normal bag

        bv, ncrops,segv,fv = inputs.shape  # inputs.shape=[64,10,32,2048], bv=batch_size*2
        bt, ncrops, segt, ft = text.shape  # text.shape=[64,10,32,300], bt=batch_size*2


        # mapping to hyperbolic space (lorentz model)
        hyp_v = self.hyp_encoder(encoder=self.lorentz, x=inputs.view(-1, segv, fv), c=self.c.exp()) 
        # print("hyp_v nan:", torch.isnan(hyp_v).any())
        hyp_t = self.hyp_encoder(encoder=self.lorentz, x=text.view(-1, segt, ft), c = self.c.exp()) 
        # print("hyp_t nan:", torch.isnan(hyp_t).any())

        # Perform dimension alignment operations here
        if hyp_v.shape[1] < hyp_t.shape[1]:  # hyp_v(vis)比out2(text)少帧
            # remove the last frame of hyp_t
            hyp_t = hyp_t[:, :(hyp_v.shape[1] - hyp_t.shape[1]), :]
        elif hyp_v.shape[1] > hyp_t.shape[1]:  # hyp_v(vis)总比out2(text)多1帧
            # padding hyp_t by repeating the last frame
            hyp_t = torch.cat((hyp_t, hyp_t[:, (hyp_t.shape[1] - hyp_v.shape[1]):, :]), dim=1)
        
        t = hyp_v.shape[1]
        ncrops = inputs.shape[1]

        # pass through multiple hyp gcn layers (visual)
        for layer in self.hyp_gcn_layers_vis:
            hde_v = self.calculate_hyperbolic_dirichlet_energy(hyp_v, c=self.c.exp())
            threshold_v = self.calculate_LSHA(hde_v, k=1)
            vis_adj = utils.hyperbolic_adj(hyp_v, threshold_v, self.c.exp(), seq_len=seq_len)  # vis_adj.shape=[640,32,32]
            gcn_input = (hyp_v, vis_adj) 
            out = self.leaky_relu(layer(gcn_input)[0])
            out = self.dropout(out)
            hyp_v = out  # hyp_v.shape=[640,32,2048]
        vis_features = hyp_v
    

        # pass through multiple hyp gcn layers (text)
        for layer in self.hyp_gcn_layers_txt:
            hde_t = self.calculate_hyperbolic_dirichlet_energy(hyp_t, c=self.c.exp())
            threshold_t = self.calculate_LSHA(hde_t, k=1)
            text_adj = utils.hyperbolic_adj(hyp_t, threshold_t, self.c.exp(), seq_len=seq_len)  # text_adj.shape=[640,32,32]
            gcn_input2 = (hyp_t, text_adj) 
            out2 = self.leaky_relu(layer(gcn_input2)[0])
            out2 = self.dropout(out2)
            hyp_t = out2  # hyp_t.shape=[640,32,768]
        text_features = hyp_t
    
        d_vis = vis_features.shape[-1]
        d_text = text_features.shape[-1]

        # transform to poincare ball model
        vis_poincare = self.hyperboloid.to_poincare(vis_features.view(-1, d_vis), c=self.c.exp()) # vis_poincare.shape=[640,32,d_vis]
        text_poincare = self.hyperboloid.to_poincare(text_features.view(-1, d_text), c=self.c.exp()) # text_poincare.shape=[640,32,d_text]
    
        vis_poincare = vis_poincare.view(-1, t, d_vis-1)  # vis_poincare.shape=[640,32,2048]
        text_poincare = text_poincare.view(-1, t, d_text-1)  # text_poincare.shape=[640,32,768]

        if self.multimodal:
            features = self.concat(vis_poincare,text_poincare, c=self.c.exp()) # features.shape = [640, 32, args.feature_size ]
        elif self.text:
            features = text_poincare # features.shape = [640, 32, 768]
        else:
            features = vis_poincare  # features.shape=[640,32,2048]
        
        features = features

        features = self.relu(self.dropout(self.hyp_liner1(features, c=self.c.exp())))

        feat_magnitudes =pm.dist0(x=features, c=self.c.exp()) # finding geodesic distance from the origin

        scores = 1 - torch.tanh(feat_magnitudes) # anomaly scores
        scores = scores.view(inputs.shape[0], inputs.shape[1], -1).mean(1) 
        scores = scores.unsqueeze(dim=2)

        # separate normal and abnormal features and scores
        normal_features = features[0:self.batch_size * ncrops]  
        normal_scores = scores[0:self.batch_size]  

        abnormal_features = features[self.batch_size * ncrops:]  
        abnormal_scores = scores[self.batch_size:]  
        
        # separate normal and abnormal feature magnitudes
        feat_magnitudes = feat_magnitudes.view(inputs.shape[0], ncrops, -1).mean(1)  
        nfea_magnitudes = feat_magnitudes[0:self.batch_size] 
        afea_magnitudes = feat_magnitudes[self.batch_size:]  

        test_features = None
       
        n_size = nfea_magnitudes.shape[0]
        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features
            test_features = self.to_euclidean(normal_features)

        #######  process abnormal videos -> select top3 feature magnitude  #######

        select_idx = torch.ones_like(nfea_magnitudes).cuda()
        select_idx = self.dropout(select_idx)
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1, largest=False)[1]  
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]]) 

        abnormal_features = abnormal_features.view(n_size, ncrops, t, -1)  
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)  

        total_select_abn_feature = torch.zeros(0)
        for abnormal_feature in abnormal_features:  
            feat_select_abn = torch.gather(abnormal_feature, 1,
                                           idx_abn_feat)  
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])  
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score),
                                    dim=1)  

        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
        select_idx_normal = self.dropout(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1, largest=False)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, -1)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1,
                                              idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)  # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature  # train: shape=[320,3,2048]
        feat_select_normal = total_select_nor_feature  # train: shape=[320,3,2048]


        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores, afea_magnitudes, nfea_magnitudes, self.c.exp(), test_features


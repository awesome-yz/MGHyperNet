import matplotlib.pyplot as plt
import torch,sys
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import numpy as np
from utils import get_gt
import os

def test(dataloader, model, args, device, step):
    """
    Implementation based on TEVAD (https://github.com/coranholmes/TEVAD/blob/main/test_10crop.py
    """
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        all_feats = np.zeros(0)
        videos_length= -1
        gt = get_gt(args.dataset, args.gt)

        for i, (input, text) in enumerate(dataloader):  # test set has 199 videos
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            text = text.to(device)
            text = text.permute(0, 2, 1, 3)

            seq_len = torch.sum(torch.max(torch.abs(input.view(-1, input.shape[-2], input.shape[-1])), dim=2)[0] > 0, 1)
            # input.shape = (1,10,T,2048); T clips, each clip has 16frames, each frame has 10 crops
            # https://github.com/tianyu0207/RTFM/issues/51

            _,_,_, _, logits, _,_, _, features = model(input, text, step, seq_len=seq_len) 
            
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            
            
            if args.save_test_results:
                features = features.squeeze(1).mean(0)
                features = features.cpu().detach().numpy()
                features = np.repeat(features, 16, axis=0)
                all_feats = np.concatenate((all_feats, features), axis=0) if all_feats.size else features
               

            #     video_pred = np.array(logits.cpu().detach().numpy())
            #     video_pred = np.repeat(video_pred, 16)
            #     videos_length += len(video_pred)

            #     v_start_index = videos_length - len(video_pred) +1
            #     v_end_index = videos_length +1
            #     # print(f'v_start_index: {v_start_index}')
            #     # print(f'v_end_index: {v_end_index}')
            #     # print(f'len(video_pred): {len(video_pred)}')
            #     video_gt = gt[v_start_index: v_end_index]
            #     # print(f'video_gt: {video_gt}')
            #     # print(f'len(video_gt): {len(video_gt)}')

            #     # print('video : ' + str(i))
            #     # print('ap : ' + str(ap))
            #     # print('auc : ' + str(rec_auc))
                
            #     path = os.path.join(args.pred, f'predictions_step_{step}', f'video_{i}')
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     np.save(os.path.join(path, 'pred.npy'), video_pred)
            #     np.save(os.path.join(path, 'gt.npy'), video_gt)
            #     np.save(os.path.join(path, 'features.npy'), features)

            pred = torch.cat((pred, sig))

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16) 
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        rec_auc = auc(fpr, tpr)
        ap = average_precision_score(list(gt), pred)

        print('auc : ' + str(rec_auc))
        print('ap : ' + str(pr_auc))

        if args.save_test_results:
            path = os.path.join(args.output_folder, f'predictions_step_{step}')
            if not os.path.exists(path):
                os.makedirs(path)
            
            np.save(os.path.join(path, 'all_pred.npy'), pred)
            np.save(os.path.join(path, 'all_gt.npy'), gt)
            np.save(os.path.join(path, 'all_features.npy'), all_feats)
            np.save(os.path.join(path, 'fpr.npy'), fpr)
            np.save(os.path.join(path, 'tpr.npy'), tpr)
            np.save(os.path.join(path, 'precision.npy'), precision)
            np.save(os.path.join(path, 'recall.npy'), recall)
            np.save(os.path.join(path, 'auc.npy'), rec_auc)
            np.save(os.path.join(path, 'ap.npy'), ap)
        return rec_auc, pr_auc


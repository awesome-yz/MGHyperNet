from torch.utils.data import DataLoader
import torch.optim as optim
from model import Model
# from improved_model import Model
from dataset import Dataset
from train import train
from test import test
from tqdm import tqdm
from utils import *
from omegacli import OmegaConf
from geoopt import optim as gpt
import argparse
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import time
import numpy as np
from evaluate_best_checkpoints import evaluate_saved_seeds
import sys


def main(args):
    """Main function to train and test the model.
    Implementation based on TEVAD (https://github.com/coranholmes/TEVAD/blob/main/main.py)
    """
    seed_everything(args.seed)
    if args.emb_folder == "":
        sb_pt_name = "vatex"
    else:
        sb_pt_name = args.emb_folder[11:]  # sent_emb_n_XXX
        
    print("Using SwinBERT pre-trained model: ", sb_pt_name)
    
    # Dataloader for normal videos
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator(device='cuda'))
    # Dataloader for abnormal videos
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator(device='cuda'))
    test_loader = DataLoader(Dataset(args, test_mode=True),
                            batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False, generator=torch.Generator(device='cuda'))

    model = Model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        print("\nMeasuring Runtime GPU Memory Footprint...")
        model.eval()
        
        # REPLICATE REAL MULTI-STREAM TRAINING SHAPE:
        # Real pipeline stacks normal and abnormal inputs -> Batch dimension must be doubled!
        real_batch_throughput = args.batch_size * 2  # FIXED: Matches your data pipeline split
        
        dummy_vis = torch.randn(real_batch_throughput, args.ncrops, 32, args.feature_size).to(device)
        dummy_text = torch.randn(real_batch_throughput, args.ncrops, 32, args.emb_dim).to(device)
        
        with torch.no_grad():
            # Pass dummy targets to satisfy sequence and shape metrics blocks
            _ = model(dummy_vis, dummy_text)
            
        # 2. Extract peak memory allocated in Megabytes (MB)
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)
        print("="*40)
        print(f"Peak Runtime GPU VRAM: {peak_memory_mb:.2f} MB")
        print("="*40 + "\n")

    
    if args.pretrained_ckpt is not None:
        print("Loading pretrained model " + args.pretrained_ckpt)
        ckpts = torch.load(args.pretrained_ckpt)
        model.load_state_dict(ckpts['model_state'])
        start_epoch = ckpts['epoch']
        optimizer.load_state_dict(ckpts['optimizer_state_dict'])
    else:
        start_epoch = 1
        if args.optimizer == 'adam':
            optimizer = gpt.RiemannianAdam(model.parameters(), lr=args.lr, weight_decay=0.0)
        elif args.optimizer == 'sgd': 
            print("Using SGD optimizer")
            momentum_val = args.momentum if args.momentum is not None else 0.9
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=momentum_val, weight_decay=args.weight_decay)

    # Define scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr) 

    test_info = {"epoch": [], "test_AUC": [], "test_AP": []}
    best_AUC, best_ap = -1, -1
    best_epoch = -1

    output_path = os.path.join(args.output_folder, f"{args.manifold}_seed_{args.seed}")
    ckpt_path = os.path.join(args.ckpt, f"{args.manifold}_seed_{args.seed}")
    
    test_time = []
    train_loss = []
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    train_st = time.time() # Record training start time
    for step in tqdm(range(start_epoch, args.epochs + 1), total=args.epochs, dynamic_ncols=True):

        if (step - 1) % len(train_nloader) == 0:  
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:  
            loadera_iter = iter(train_aloader)

        loss = train(loadern_iter, loadera_iter, model, args, optimizer, device, step, train_loss)
        scheduler.step() # Progress learning rate decay smoothly

        if step % 5 == 0 and step > 50:
            test_time_st = time.time()

            auc, ap = test(test_loader, model, args, device, step)
            test_time_end = time.time()
            test_time.append(test_time_end - test_time_st)
            print(f'test time for epoch {step}: {test_time_end - test_time_st:.2f}s')
            print(f'epoch: {step}, test_auc: {auc}, test_ap: {ap}')
        

            if "violence" in args.dataset:  
                if test_info["test_AP"][-1] > best_ap:
                    best_ap = test_info["test_AP"][-1]
                    best_epoch = step
                    checkpoint_data = {
                        'epoch': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }
                    torch.save(checkpoint_data, os.path.join(ckpt_path, f"{args.dataset}_best_ap.pkl"))
                    save_best_record(test_info, os.path.join(output_path, f"{args.dataset}_{step}_{sb_pt_name}-AP.txt"), "test_AP")

                APs = test_info["test_AP"]
                APs_mean, APs_median, APs_std, APs_max, APs_min = np.mean(APs), np.median(APs), np.std(APs), np.max(APs), np.min(APs)
                print("std\tmean\tmedian\tmin\tmax\tAP")
                print(f"{APs_std * 100:.2f}\t{APs_mean * 100:.2f}\t{APs_median * 100:.2f}\t{APs_min * 100:.2f}\t{APs_max * 100:.2f}")

                if test_info["test_AUC"][-1] > best_AUC:
                    best_AUC = test_info["test_AUC"][-1]
                    best_epoch = step
                    checkpoint_data = {
                        'epoch': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }
                    torch.save(checkpoint_data, os.path.join(ckpt_path, f"{args.dataset}_best_auc.pkl"))
                    save_best_record(test_info, os.path.join(output_path, f"{args.dataset}_{step}_{sb_pt_name}-AUC.txt"), "test_AUC")

                AUCs = test_info["test_AUC"]
                AUCs_mean, AUCs_median, AUCs_std, AUCs_max, AUCs_min = np.mean(AUCs), np.median(AUCs), np.std(AUCs), np.max(AUCs), np.min(AUCs)
                print("std\tmean\tmedian\tmin\tmax\tAUC")
                print(f"{AUCs_std * 100:.2f}\t{AUCs_mean * 100:.2f}\t{AUCs_median * 100:.2f}\t{AUCs_min * 100:.2f}\t{AUCs_max * 100:.2f}")
            
            else: 
                test_info["epoch"].append(step)
                test_info["test_AUC"].append(auc)
                test_info["test_AP"].append(ap)
                
                if test_info["test_AUC"][-1] > best_AUC:
                    best_AUC = test_info["test_AUC"][-1]
                    best_epoch = step
                    checkpoint_data = {
                        'epoch': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }
                    torch.save(checkpoint_data, os.path.join(ckpt_path, f"{args.dataset}_best_auc.pkl"))
                    save_best_record(test_info, os.path.join(output_path, f"{args.dataset}_{step}_{sb_pt_name}-AUC.txt"), "test_AUC")

                APs = test_info["test_AP"]
                APs_mean, APs_median, APs_std, APs_max, APs_min = np.mean(APs), np.median(APs), np.std(APs), np.max(APs), np.min(APs)
                print("std\tmean\tmedian\tmin\tmax\tAP")
                print(f"{APs_std * 100:.2f}\t{APs_mean * 100:.2f}\t{APs_median * 100:.2f}\t{APs_min * 100:.2f}\t{APs_max * 100:.2f}")
                
                AUCs = test_info["test_AUC"]
                AUCs_mean, AUCs_median, AUCs_std, AUCs_max, AUCs_min = np.mean(AUCs), np.median(AUCs), np.std(AUCs), np.max(AUCs), np.min(AUCs)
                print("std\tmean\tmedian\tmin\tmax\tAUC")
                print(f"{AUCs_std * 100:.2f}\t{AUCs_mean * 100:.2f}\t{AUCs_median * 100:.2f}\t{AUCs_min * 100:.2f}\t{AUCs_max * 100:.2f}")
    
    print(f"\nTraining Complete! Best performance at Epoch {best_epoch} with AUC: {best_AUC:.4f}")
    with open(os.path.join(args.output_folder, 'seeds_evaluation.txt'), '+a') as f:
        f.write(f'seed:{args.seed}, AP:{APs_max:.4f}, AUC:{AUCs_max:.4f}\n')
        f.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml', help='path to the configuration')
    parser.add_argument('--seed', type=int, default=4869, help='Random initialization seed')
    cmd_args = parser.parse_args()

    with open(cmd_args.config, 'r') as f:
        args = OmegaConf.load(f)
    
    if '--seed' in sys.argv:
        args.seed = cmd_args.seed
        print(f"--> [MHyperNet] Terminal override: Forcing active seed to {args.seed}")
   
    main(args)
    if args.evaluate_seeds:
        evaluate_saved_seeds(args)

   
    
   

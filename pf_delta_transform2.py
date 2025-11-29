import torch
import numpy as np
import argparse
import sys
import os

# Add path to the PFDelta code
sys.path.append(os.getcwd()) 
from core.datasets.pfdelta_dataset import PFDeltaDataset

def convert_dataset(root_dir, output_file, case="case14", n_samples=1000):
    print(f"--- Initializing PFDelta Dataset Conversion ({case}) ---")
    print(f"Target Structure: [Pd, Pg, Qd, Qg, Vm, Va, TypePQ, TypePV, TypeSlack]")
    
    tokens_list = []
    
    splits = ["train"]
    
    # We load a small subset if we just want to verify, but for training load more
    # Adjust n_samples as needed for your full training
    #SAMPLES_PER_SPLIT = 1000
    
    for split_name in splits:
        try:
            print(f"Loading {split_name}...")
            dataset = PFDeltaDataset(
                root_dir=root_dir, case_name=case, perturbation="n", 
                split=split_name, model="CANOS", task=1.1,
                add_bus_type=True, force_reload=False,
                n_samples=1000
            )
        except Exception as e:
            print(f"Skipping {split_name}: {e}")
            continue

        if len(dataset) == 0:
            print(f"Warning: {split_name} is empty.")
            continue

        for i in range(len(dataset)):
            data = dataset[i]
            
            # --- 1. Extract Raw Features ---
            # PowerModels/PyG data['bus'] features:
            # bus_gen: [Pg, Qg]
            # bus_demand: [Pd, Qd]
            # bus_voltages: [Va, Vm]  <-- Note order in dataset class is usually [Va, Vm]
            
            pg = data['bus'].bus_gen[:, 0]
            qg = data['bus'].bus_gen[:, 1]
            pd = data['bus'].bus_demand[:, 0]
            qd = data['bus'].bus_demand[:, 1]
            
            va = data['bus'].bus_voltages[:, 0]
            vm = data['bus'].bus_voltages[:, 1]
            
            # --- 2. Bus Types (One-Hot) ---
            # Raw types: 1=PQ, 2=PV, 3=Slack (usually)
            # PyG implementation in dataset often converts to 0-based or keeps as is.
            # Let's inspect min value to be safe.
            b_type = data['bus'].bus_type.long()
            if b_type.min() >= 1: 
                b_type -= 1 
            # Clamp to 0-2 range just in case (0:PQ, 1:PV, 2:Slack)
            b_type = torch.clamp(b_type, 0, 2)
            
            # One-hot encoding: [N, 3]
            b_type_oh = torch.nn.functional.one_hot(b_type, num_classes=3).float()
            
            # --- 3. Construct Unified Token ---
            # Channels:
            # 0: Pd
            # 1: Pg
            # 2: Qd
            # 3: Qg
            # 4: Vm
            # 5: Va
            # 6: TypePQ
            # 7: TypePV
            # 8: TypeSlack
            
            token = torch.stack([
                pd, pg, qd, qg, vm, va
            ], dim=1) # [N, 6]
            
            # Concat with One-Hot Types -> [N, 9]
            token = torch.cat([token, b_type_oh], dim=1)
            
            tokens_list.append(token.numpy())

            if (i+1) % 1000 == 0:
                print(f"Processed {i+1} samples in {split_name}")
            
            if(i == 999):
                break

    if not tokens_list:
        print("No data loaded!")
        return

    all_tokens = np.array(tokens_list)
    
    print(f"\n--- Final Data Check ---")
    print(f"Tokens Shape: {all_tokens.shape}")
    print(f"Channels: 0:Pd, 1:Pg, 2:Qd, 3:Qg, 4:Vm, 5:Va, 6-8:Types")
    print(f"Pd Mean: {all_tokens[:,:,0].mean():.4f}, Std: {all_tokens[:,:,0].std():.4f}")
    print(f"Pg Mean: {all_tokens[:,:,1].mean():.4f}, Std: {all_tokens[:,:,1].std():.4f}")
    print(f"Qd Mean: {all_tokens[:,:,2].mean():.4f}, Std: {all_tokens[:,:,2].std():.4f}")
    print(f"Qg Mean: {all_tokens[:,:,3].mean():.4f}, Std: {all_tokens[:,:,3].std():.4f}")
    print(f"Vm Mean: {all_tokens[:,:,4].mean():.4f}, Std: {all_tokens[:,:,4].std():.4f}")
    print(f"Va Mean: {all_tokens[:,:,5].mean():.4f}, Std: {all_tokens[:,:,5].std():.4f}")
    
    np.savez_compressed(output_file, tokens=all_tokens)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="pfdelta_unified.npz")
    parser.add_argument("--case", type=str, default="case14")   
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()
    
    convert_dataset(args.root_dir, args.output, args.case, args.n_samples)
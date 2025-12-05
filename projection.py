# import required libraries
import torch
import torch.nn as nn
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import time
import argparse
from pathlib import Path
import json
import copy

# indexing
IDX_PD, IDX_PG = 0, 1
IDX_QD, IDX_QG = 2, 3
IDX_VM, IDX_VA = 4, 5
IDX_TYPE_PQ, IDX_TYPE_PV, IDX_TYPE_SLACK = 6, 7, 8
NORM=[IDX_PD, IDX_QD, IDX_PG, IDX_QG]

# model
class MAEModel(nn.Module):
    def __init__(self, token_dim, embed_dim, num_buses, num_layers=4):
        super().__init__()
        self.token_embedding = nn.Linear(token_dim, embed_dim)
        self.bus_id_embedding = nn.Embedding(num_buses, embed_dim)
        
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, token_dim)
        )
    def forward(self, x, node_ids):
        x_emb = self.token_embedding(x)
        topo_emb = self.bus_id_embedding(node_ids)
        x_in = x_emb + topo_emb
        x_enc = self.encoder(x_in)
        out = self.decoder(x_enc)
        return out

# load pandapower network for parameters
def load_pandapower_network(case_name):
    case_map = {
        'case14': pn.case14,
        'case30': pn.case30,
        'case57': pn.case57,
        'case118': pn.case118,
    }
    
    if case_name not in case_map:
        raise ValueError(f"Unknown case: {case_name}. Available: {list(case_map.keys())}")    
    return case_map[case_name]()

# token -> network
def token_to_pandapower(net, token):
    n_buses = len(net.bus)
    
    net.load.drop(net.load.index, inplace=True)
    net.gen.drop(net.gen.index, inplace=True)
    net.ext_grid.drop(net.ext_grid.index, inplace=True)

    slack_count=0

    for bus_idx in range(n_buses):
        pd = token[bus_idx, IDX_PD]
        pg = token[bus_idx, IDX_PG]
        qd = token[bus_idx, IDX_QD]
        qg = token[bus_idx, IDX_QG]
        vm = token[bus_idx, IDX_VM]
        va_radians = token[bus_idx, IDX_VA]
        va_degrees = np.degrees(va_radians)
        
        is_pq = token[bus_idx, IDX_TYPE_PQ] > 0.5
        is_pv = token[bus_idx, IDX_TYPE_PV] > 0.5
        is_slack = token[bus_idx, IDX_TYPE_SLACK] > 0.5
       
        if is_slack:
            pp.create_ext_grid(net, bus=bus_idx, vm_pu=vm, va_degree=va_degrees)
            slack_count += 1

        elif is_pv:
#            if abs(pg) > 1e-6:
                pp.create_gen(net, bus=bus_idx, p_mw=pg, vm_pu=vm)

        elif is_pq and abs(pg) > 1e-6:
             pp.create_sgen(net, bus=bus_idx, p_mw=pg, q_mvar=qg)
        
        if abs(pd) > 1e-6 or abs(qd) > 1e-6:
            pp.create_load(net, bus=bus_idx, p_mw=pd, q_mvar=qd)

    if slack_count == 0:
        raise ValueError("No slack bus found in token data!")
    if slack_count > 1:
        print(f"Warning: Multiple slack buses found ({slack_count})")        
        
# model + NR
def run_model_plus_nr(net_template, token_input, model, device, stats):
    # Input is not normalised token
    token_norm = (token_input - stats['means']) / stats['stds']
    mask = np.zeros_like(token_norm, dtype=bool)

    is_pq = token_norm[:, IDX_TYPE_PQ] > 0.5
    is_pv = token_norm[:, IDX_TYPE_PV] > 0.5
    is_slack = token_norm[:, IDX_TYPE_SLACK] > 0.5

    mask[:, IDX_VM] = is_pq
    mask[:, IDX_VA] = is_pq | is_pv
    mask[:, IDX_QG] = is_pv | is_slack
    mask[:, IDX_PG] = is_slack

    token_masked = token_norm.copy()
    token_masked[mask] = 0.0

    # Model inference
    token_tensor = torch.from_numpy(token_masked).unsqueeze(0).to(device)
    node_ids = torch.arange(len(token_input)).unsqueeze(0).to(device)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        pred_norm = model(token_tensor, node_ids)
    t_model = time.perf_counter() - t0
    
    # Get prediction
    pred_norm_np = pred_norm.cpu().numpy()[0]

    # De-normalize
    pred = (pred_norm_np * stats['stds']) + stats['means']

    vm_pred = pred[:, IDX_VM] 
    va_pred = np.rad2deg(pred[:, IDX_VA])

    net = copy.deepcopy(net_template)
    token_to_pandapower(net, token_input) 
    
    # Run NR from model's warm start
    t0 = time.perf_counter()
    try:
        
        pp.runpp(net, init='auto', algorithm='nr', init_vm_pu=vm_pred, init_va_degree=va_pred, calculate_voltage_angles=True, enforce_q_lims=True, numba=False, max_iteration=50)
        converged = net.converged
        iters = net._ppc["iterations"]
        if converged:
            print(f"PF (NR+Model) converged in {iters} iterations.")
            # To check for results
            #print("\nBus Results:")
            #print(net.res_bus)
    except Exception as e:
         print(f"Model+NR failed: {e}")
         converged = False
    t_nr = time.perf_counter() - t0
    
    return t_model, t_nr, t_model + t_nr, converged

def run_nr_only(net_template, token_input):
    # Setup network with input data
    net = copy.deepcopy(net_template)
    token_to_pandapower(net, token_input)
    
    # Run NR from flat start
    t0 = time.perf_counter()
    try:
        pp.runpp(net, init='flat', algorithm='nr',calculate_voltage_angles=True, enforce_q_lims=True, numba=False, max_iteration=50)
        converged = net.converged
        iters = net._ppc["iterations"]
        if converged:
            print(f"PF(NR) converged in {iters} iterations.")
            # To check for results
            #print("\nBus Results:")
            #print(net.res_bus)
    except Exception as e:
        print(f"NR-only failed: {e}")
        converged = False
    t_nr = time.perf_counter() - t0
    
    return t_nr, converged

def benchmark_case(case_name, model_path, data_path, n_samples, device, embed_dim, num_layers):

    print(f"\n{'='*60}")
    print(f"Benchmarking {case_name.upper()}")
    print(f"{'='*60}")
    
    # Load network template
    net_template = load_pandapower_network(case_name)
    n_buses = len(net_template.bus)
    
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    data = np.load(data_path)
    all_tokens = data['tokens']  
    
    print(f"Dataset shape: {all_tokens.shape}")
    print(f"Total samples available: {len(all_tokens)}")
    

    if len(all_tokens) > n_samples:
        tokens_to_use = all_tokens[:n_samples]
        print(f"Using first {n_samples} samples out of {len(all_tokens)}")
    else:
        tokens_to_use = all_tokens
        n_samples = len(tokens_to_use)
        print(f"Using all {n_samples} samples")

    # means = np.zeros(9, dtype=np.float32)
    # stds = np.ones(9, dtype=np.float32)   
    # train_flat = tokens_to_use.reshape(-1, 9)
    # for c in NORM:
    #     means[c] = train_flat[:, c].mean()
    #     stds[c] = train_flat[:, c].std() + 1e-8
    # stats = {'means': means, 'stds': stds}
    
    # Normalization
    stats_path = Path(model_path).parent / "normalization_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found at {stats_path}. "
    )
    print(f"Loading normalization stats from {stats_path}...")
    stats_data = np.load(stats_path)
    means = stats_data['mean']
    stds = stats_data['std']
    stats = {'means': means, 'stds': stds}


    # Load model
    print(f"Loading model for {n_buses} buses...")
    model = MAEModel(
        token_dim=9,
        embed_dim=embed_dim,
        num_buses=n_buses,
        num_layers=num_layers
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Benchmark
    print(f"Running benchmark on {n_samples} samples from dataset...")    
    results = {
        'model_time': [],
        'model_nr_time': [],
        'model_total_time': [],
        'nr_only_time': [],
        'converged_model_nr': [],
        'converged_nr_only': []
    }
    
    for i in range(n_samples):
        token_orig = tokens_to_use[i]        
        # Model + NR
        try:
            t_model, t_nr, t_total, conv = run_model_plus_nr(
                net_template, token_orig, model, device, stats
            )
            if conv:
                results['model_time'].append(t_model)
                results['model_nr_time'].append(t_nr)
                results['model_total_time'].append(t_total)
            results['converged_model_nr'].append(conv)
            
        except Exception as e:
            print(f"Sample {i} Model+NR failed: {e}")
            results['converged_model_nr'].append(False)
        
        # NR only
        try:
            t_nr_only, conv = run_nr_only(net_template, token_orig)
            if conv:
                results['nr_only_time'].append(t_nr_only)
            results['converged_nr_only'].append(conv)
        except Exception as e:
            print(f"Sample {i} NR-only failed: {e}")
            results['converged_nr_only'].append(False)
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{n_samples} samples")   

    model_nr_times = np.array(results['model_total_time']) if len(results['model_total_time'])>0 else np.array([0])
    nr_only_times = np.array(results['nr_only_time']) if len(results['nr_only_time'])>0 else np.array([0])
    n_success_model_nr = sum(results['converged_model_nr']) 
    n_success_nr_only = sum(results['converged_nr_only'])
    
    print(f"\n{'-'*60}")
    print(f"RESULTS FOR {case_name.upper()} ({n_buses} buses)")
    print(f"{'-'*60}")
    print(f"Samples: {n_samples} total, Model+NR: {n_success_model_nr} converged, NR-only: {n_success_nr_only} converged")
    print(f"Model Inference Time:  {np.mean(results['model_time'])*1000:.2f}ms (avg over {len(results['model_time'])} successful)")
    print(f"Model+NR NR Time:      {np.mean(results['model_nr_time'])*1000:.2f}ms (avg over {len(results['model_nr_time'])} successful)")
    print(f"Model+NR Total Time:   {np.mean(model_nr_times)*1000:.2f}ms (avg over {len(model_nr_times)} successful)")
    print(f"NR-only Time:          {np.mean(nr_only_times)*1000:.2f}ms (avg over {len(nr_only_times)} successful)")
    if len(model_nr_times) > 0 and len(nr_only_times) > 0:
        print(f"Speedup Factor:        {np.mean(nr_only_times)/np.mean(model_nr_times):.2f}x")
    else:
        print(f"Speedup Factor:        N/A (insufficient successful runs)")
    print(f"Model+NR Convergence:  {n_success_model_nr}/{n_samples} ({100*n_success_model_nr/n_samples:.1f}%)")
    print(f"NR-only Convergence:   {n_success_nr_only}/{n_samples} ({100*n_success_nr_only/n_samples:.1f}%)")

    return {
        'case': case_name,
        'n_buses': n_buses,
        'n_samples': n_samples,
        'n_success_model_nr': n_success_model_nr,
        'n_success_nr_only': n_success_nr_only,
        'model_time_ms': float(np.mean(results['model_time']) * 1000) if results['model_time'] else None,
        'model_nr_time_ms': float(np.mean(results['model_nr_time']) * 1000) if results['model_nr_time'] else None,
        'model_nr_total_ms': float(np.mean(model_nr_times) * 1000) if len(model_nr_times) > 0 else None,
        'nr_only_ms': float(np.mean(nr_only_times) * 1000) if len(nr_only_times) > 0 else None,
        'speedup': float(np.mean(nr_only_times) / np.mean(model_nr_times)) if len(model_nr_times) > 0 and len(nr_only_times) > 0 else None,
        'model_nr_convergence': n_success_model_nr / n_samples,
        'nr_only_convergence': n_success_nr_only / n_samples,
}

def main():
    parser = argparse.ArgumentParser(description='Benchmark Model+NR vs NR-only')
    parser.add_argument('--case', type=str, required=True,
                        help='Case to benchmark')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data file')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of test samples to benchmark')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Model embedding dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for results')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    result = benchmark_case(
        case_name=args.case,
        model_path=args.model,
        data_path=args.data,
        n_samples=args.n_samples,
        device=device,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
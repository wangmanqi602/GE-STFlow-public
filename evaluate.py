"""GE-STFlow 评估, 指标口径与原版 eval.py 一致"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from dataset import build_datasets, make_loaders
from models.gestflow import GESTFlow

@torch.no_grad()
def evaluate(model, test_loader, config, device, num_steps=32, method="ode", mean=0.0, std=1.0,
             seed: int = 0):
    model.eval()
    preds, tgts = [], []
    t0 = time.time()
    for i, batch in enumerate(test_loader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred = model.sample(batch["sst_fields"], batch["coords"], batch["target_idx"],
                            batch["land_mask"], batch["doys"],
                            num_steps=num_steps, p_mean=config.get("p_mean", 0.0),
                            seed=seed)
        preds.append(pred.cpu()); tgts.append(batch["target_trajectory"].cpu())
        if (i+1)%50==0: print(f"  eval batch {i+1}/{len(test_loader)} ({time.time()-t0:.0f}s)", flush=True)
    pred = torch.cat(preds); tgt = torch.cat(tgts)
    mse = torch.mean((pred-tgt)**2).item(); rmse = mse**0.5; mae = torch.mean(torch.abs(pred-tgt)).item()
    pred_raw = pred*std+mean; tgt_raw = tgt*std+mean
    mae_raw = torch.mean(torch.abs(pred_raw-tgt_raw)).item()
    mask = torch.abs(tgt_raw)>1.0
    mape = (torch.abs((pred_raw[mask]-tgt_raw[mask])/tgt_raw[mask])*100).mean().item() if mask.sum()>0 else 0.0
    plist = []
    for b in range(pred.shape[0]):
        p,t = pred[b],tgt[b]
        if p.std()>1e-8 and t.std()>1e-8: plist.append(((p-p.mean())*(t-t.mean())).mean()/(p.std()*t.std()).item())
    pearson = float(np.mean(plist)) if plist else float("nan")
    ne = pred_raw.numel()
    psum,psq = pred_raw.sum().item(),(pred_raw**2).sum().item()
    tsum,tsq = tgt_raw.sum().item(),(tgt_raw**2).sum().item()
    sp = (psq/ne-(psum/ne)**2)**0.5; so = (tsq/ne-(tsum/ne)**2)**0.5
    sstot = tsq-tsum**2/ne; ssres = ((pred_raw-tgt_raw)**2).sum().item()
    r2 = 1.0-ssres/sstot if sstot>0 else float("nan")
    e = (pred_raw-tgt_raw).numpy()
    return {"mse":mse,"rmse":rmse,"mae":mae,"mae_raw(°C)":mae_raw,"mape(%)":mape,
            "pearson_r":pearson,"r2":r2,"sigma_pred(°C)":sp,"sigma_obs(°C)":so,"n_samples":pred.shape[0],
            "rmse_per_lead":(e**2).mean(0)**0.5,"mae_per_lead":np.abs(e).mean(0),
            "pred":pred,"tgt":tgt}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--out", default="checkpoints/stidiff_v2")
    p.add_argument("--num_steps", type=int, default=32); p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0, help="采样随机种子 (固定以复现)")
    p.add_argument("--save_export", action="store_true"); args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrs, mean, std = build_datasets()
    _,_,tl = make_loaders(arrs,args.batch_size,num_workers=4,drop_last_test=True)
    ckpt = torch.load(args.ckpt,map_location="cpu",weights_only=False)
    cfg = ckpt.get("config",{})
    model = GESTFlow(
        d_model=cfg.get("d_model",256), n_heads=cfg.get("n_heads",8),
        n_encoder_layers=cfg.get("n_encoder_layers",4),
        n_denoiser_layers=cfg.get("n_denoiser_layers",4),
        T_in=cfg.get("T_in",30), T_out=cfg.get("T_out",60),
        n_time_tokens=cfg.get("n_time_tokens",4),
        noise_scale=cfg.get("noise_scale",1.0), t_eps=cfg.get("t_eps",5e-2),
        use_residual=cfg.get("use_residual",False),
    ).to(device)
    sd = ckpt.get("ema_state_dict",ckpt.get("model_state_dict"))
    model.load_state_dict({k:torch.tensor(v) for k,v in sd.items()})
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch','?')})")
    st = evaluate(model,tl,cfg,device,num_steps=args.num_steps,mean=mean,std=std,seed=args.seed)
    print(f"\n{'='*72}")
    print(f"Test | RMSE={st['rmse']:.6f} | MAE_raw={st['mae_raw(°C)']:.4f}°C | MAPE={st['mape(%)']:.2f}% | Pearson={st['pearson_r']:.4f} | R²={st['r2']:.4f} | σ_pred={st['sigma_pred(°C)']:.4f} | σ_obs={st['sigma_obs(°C)']:.4f} | N={st['n_samples']}")
    print(f"{'='*72}")
    out_dir = Path(args.out); out_dir.mkdir(parents=True,exist_ok=True)
    summary = {k:v for k,v in st.items() if not torch.is_tensor(v) and not isinstance(v,np.ndarray)}
    summary.update({"rmse_per_lead":st["rmse_per_lead"].tolist(),"mae_per_lead":st["mae_per_lead"].tolist(),
                    "num_steps":args.num_steps,"mean":mean,"std":std})
    with open(out_dir/"eval_results.json","w") as f: json.dump(summary,f,indent=2)
    print(f"保存: {out_dir/'eval_results.json'}")
    if args.save_export:
        ex = {"pred":st["pred"].numpy(),"tgt":st["tgt"].numpy(),
              "error":(st["pred"]*std+mean-st["tgt"]*std+mean).numpy(),
              "mean":mean,"std":std,"mae_per_lead":st["mae_per_lead"],"rmse_per_lead":st["rmse_per_lead"]}
        torch.save(ex, out_dir/"test_export.pt",_use_new_zipfile_serialization=False)

if __name__=="__main__": main()

"""Evaluate DatasetHypernet: per-dataset metrics (functional MSE, denoising gain, toeplitzness)."""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from src.data.dataset import conv1d_same, generate_dataset
from src.evaluation._shared import config_dict_to_datasetconfig
from src.evaluation.evaluate import kernel_recovery, toeplitzness
from src.models.hypernetwork import functional_forward
from src.models.weight_codec import WeightCodec
from src.models.dataset_hypernet import DatasetEncoder, WeightDecoder, DatasetHypernet
from src.training.train_dataset_hypernet import DatasetHypernetConfig
from src.data.corpus_loader import config_from_record, load_relu_corpus
from src.data.dataset import DatasetFamily

def _resolve_device(d):
    return torch.device("cuda" if d=="auto" and torch.cuda.is_available() else d)

def evaluate(args):
    device = _resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = DatasetHypernetConfig(**ckpt["config"])
    H = cfg.mlp_hidden
    D = WeightCodec(L=32, H=H).D
    codec = WeightCodec(L=32, H=H)

    encoder = DatasetEncoder(L=32, K_enc=cfg.K_enc, d_model=cfg.d_model,
                             d_emb=cfg.d_emb, n_layers=cfg.n_layers, n_heads=cfg.n_heads)
    decoder = WeightDecoder(d_emb=cfg.d_emb, D=D)
    model = DatasetHypernet(encoder, decoder, mlp_hidden=H).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    bundle = load_relu_corpus(corpus_dir=args.corpus_dir, val_configs=cfg.val_configs, seed=cfg.seed)
    print(f"[eval_dh] {bundle.n_configs} datasets, D={D}")

    cache = {}
    family = DatasetFamily(n_test=512, L=32)
    for i in range(bundle.n_configs):
        cfg_d = config_from_record(bundle.configs[i])
        inst = family.sample_dataset(cfg_d)
        cache[i] = (inst.x_train.float().to(device), inst.y_train.float().to(device))

    def _eval(dids):
        results = []
        for did in dids:
            cfg_d = bundle.configs[did]
            ds_cfg = config_dict_to_datasetconfig(cfg_d)
            _, _, x_test, y_test, kernel, _ = generate_dataset(ds_cfg)
            x_test, y_test = x_test.to(device), y_test.to(device)
            theta_in = bundle.weights[did].to(device)
            n_mlp = theta_in.shape[0]

            x_enc = cache[did][0][:cfg.K_enc].unsqueeze(0)
            y_enc = cache[did][1][:cfg.K_enc].unsqueeze(0)

            with torch.no_grad():
                y_in = functional_forward(theta_in, x_test.unsqueeze(0).expand(n_mlp, -1, -1), L=32, H=H)
                theta_out = model.forward(x_enc, y_enc)
                y_gen = functional_forward(theta_out, x_test.unsqueeze(0), L=32, H=H)
                k_arr = np.asarray(cfg_d["kernel"], dtype=np.float32)
                y_clean = torch.from_numpy(conv1d_same(x_test.cpu().numpy().astype(np.float32), k_arr)).to(device)

            y_tgt = y_test.unsqueeze(0).expand(n_mlp, -1, -1)
            mse_gen = float(torch.mean((y_gen - y_tgt)**2).item())
            mse_in = float(torch.mean((y_in - y_tgt)**2).item())
            mse_oracle = float(torch.mean((y_clean - y_tgt)**2).item())
            y_clean_b = y_clean.unsqueeze(0).expand(n_mlp, -1, -1)
            clean_gen = float(torch.mean((y_gen - y_clean_b)**2).item())
            clean_in = float(torch.mean((y_in - y_clean_b)**2).item())
            params = codec.unpack(theta_out[0].cpu())
            M = (params["fc2.weight"] @ params["fc1.weight"]).cpu()
            toe = toeplitzness(M)
            krec = kernel_recovery(M, kernel)
            results.append({"dataset": did, "family": str(cfg_d["family"]), "radius": int(cfg_d["radius"]),
                 "f_gen": mse_gen, "f_in": mse_in, "oracle": mse_oracle,
                 "clean_gen": clean_gen, "clean_in": clean_in,
                 "ratio": mse_gen/mse_oracle if mse_oracle>0 else float("inf"),
                 "gain": clean_in/clean_gen if clean_gen>0 else float("inf"),
                 "toeplitzness": toe, "kernel_cosine": krec["cosine_sim"], "kernel_l2": krec["l2_dist"]})
        return results

    def _summary(results, tag):
        def _s(vals):
            vv=[float(v) for v in vals if np.isfinite(v)]
            return {"mean":float(np.mean(vv)),"std":float(np.std(vv)),"n":len(vv)} if vv else {"mean":float("nan"),"std":float("nan"),"n":0}
        s={"f_gen":_s([r["f_gen"] for r in results]),"f_in":_s([r["f_in"] for r in results]),
           "ratio":_s([r["ratio"] for r in results]),"gain":_s([r["gain"] for r in results]),
           "toeplitzness":_s([r["toeplitzness"] for r in results]),
           "per_dataset": results}
        print(f"\n[eval_dh] {tag.upper()} (n={len(results)}): f_gen={s['f_gen']['mean']:.3f} f_in={s['f_in']['mean']:.3f} ratio={s['ratio']['mean']:.1f} gain={s['gain']['mean']:.3f} toe={s['toeplitzness']['mean']:.3f}")
        return s

    held_out = _summary(_eval(bundle.val_cfg_indices), "HELD-OUT")
    train = _summary(_eval(bundle.train_cfg_indices), "TRAIN")

    summary={"held_out": {k:v for k,v in held_out.items() if k!="per_dataset"},
             "train": {k:v for k,v in train.items() if k!="per_dataset"},
             "held_out_per_dataset": held_out["per_dataset"],
             "train_per_dataset": train["per_dataset"]}
    out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    with open(os.path.join(out_dir, "dh_summary.json"),"w") as f: json.dump(summary, f, indent=2)
    print(f"[eval_dh] summary -> {out_dir}/dh_summary.json")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus-dir", default="data/processed/targets_relu_h16")
    p.add_argument("--device", default="cuda")
    evaluate(p.parse_args())

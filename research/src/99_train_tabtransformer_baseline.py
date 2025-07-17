#!/usr/bin/env python
"""
99_train_tabtransformer_baseline.py
----------------------------------
TabTransformer による 'ret_2' 回帰モデル（fold-0 データ）。

・GPU: 2×RTX A6000 → torch.nn.DataParallel で自動活用
・カテゴリ列は dtype == object/category または
  「ユニーク数 <= 200 & dtype は整数」の列とみなす
・数値列は StandardScaler で Z-score 正規化し、平均/分散を
  TabTransformer に渡して内部で LayerNorm する
・EarlyStopping(patience=20)＋学習曲線保存
・バリデーション／テストで RMSE, R², 符号的中率を計算
"""

import os, math, warnings, random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tab_transformer_pytorch import TabTransformer   # pip install tab-transformer-pytorch

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# ------------------------- 1. Utility ----------------------------------------
RNG_SEED = 42
def set_seed(seed=RNG_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rmse(pred, true):                   # pred & true: torch.Tensor
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()

# ------------------------- 2. Dataset ----------------------------------------
class TabularDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, target, cat_maps, scaler):
        self.y  = torch.tensor(df[target].values, dtype=torch.float32).unsqueeze(1)
        # categorical → integer codes 0…n-1
        cats = []
        for col, mapping in cat_maps.items():
            codes = df[col].map(mapping).fillna(0).astype(int).values
            cats.append(torch.tensor(codes, dtype=torch.long))
        self.x_cat = torch.stack(cats, dim=1) if cats else torch.empty(len(df),0).long()
        # continuous → z-score
        cont = torch.tensor(df[cont_cols].values, dtype=torch.float32)
        cont = (cont - scaler["mean"]) / scaler["std"]
        self.x_cont = cont
    def __len__(self):  return len(self.y)
    def __getitem__(self,i):  return self.x_cat[i], self.x_cont[i], self.y[i]

# ------------------------- 3. Main -------------------------------------------
def main():
    set_seed()

    # ------------------ 3-1. Load data ---------------------------------------
    data_dir = Path("/app/ml_bot/tmp/fold0")
    df_all  = pd.read_parquet(data_dir/"train.parquet")
    df_test = pd.read_parquet(data_dir/"test.parquet")

    # 列除外 & フィルタリング
    target = "ret_2"
    drop_targets = ["vol_2","mdd_2","ret_4","vol_4","mdd_4","ret_6","vol_6","mdd_6"]
    feature_cols = [c for c in df_all.columns if c not in [target]+drop_targets]
    for df in (df_all, df_test):
        df.dropna(subset=[target], inplace=True)
        df.query("signal != 0", inplace=True)

    # --------------- 3-2. Identify categorical / continuous ------------------
    candidate_cat = [c for c in feature_cols
                     if (df_all[c].dtype=="object") or
                        (str(df_all[c].dtype).startswith("category")) or
                        (pd.api.types.is_integer_dtype(df_all[c]) and
                         df_all[c].nunique()<=200)]
    cat_cols  = sorted(candidate_cat)
    cont_cols = sorted(list(set(feature_cols) - set(cat_cols)))

    # category → code マップ（0 reserved for NaN/UNK）
    cat_maps = {}
    categories = []
    for c in cat_cols:
        uniq = df_all[c].dropna().unique()
        mapping = {v:i+1 for i,v in enumerate(sorted(uniq))}
        cat_maps[c] = mapping           # str/int → idx
        categories.append(len(mapping)+1) # +1 for UNK

    # continuous scaler
    cont_mean = df_all[cont_cols].mean().values
    cont_std  = df_all[cont_cols].std(ddof=0).replace(0,1e-9).values
    scaler = {"mean": torch.tensor(cont_mean, dtype=torch.float32),
              "std" : torch.tensor(cont_std , dtype=torch.float32)}

    # --------------- 3-3. Split train/valid ----------------------------------
    df_train, df_valid = train_test_split(df_all, test_size=0.2, random_state=RNG_SEED)

    # --------------- 3-4. Build loaders --------------------------------------
    bs = 1024
    train_ds = TabularDataset(df_train,cat_cols,cont_cols,target,cat_maps,scaler)
    valid_ds = TabularDataset(df_valid,cat_cols,cont_cols,target,cat_maps,scaler)
    test_ds  = TabularDataset(df_test ,cat_cols,cont_cols,target,cat_maps,scaler)

    train_loader = DataLoader(train_ds,batch_size=bs,shuffle=True,num_workers=4,pin_memory=True)
    valid_loader = DataLoader(valid_ds,batch_size=bs*2,shuffle=False,num_workers=4,pin_memory=True)
    test_loader  = DataLoader(test_ds ,batch_size=bs*2,shuffle=False,num_workers=4,pin_memory=True)

    # --------------- 3-5. Build TabTransformer -------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shared_embed = bool(cat_cols)
    tt = TabTransformer(
        categories            = tuple(categories),     # ユニーク数 per cat
        num_continuous        = len(cont_cols),
        dim                   = 64,     # ↑32 も可。A6000 なので余裕を持たせ 64
        dim_out               = 1,      # 回帰
        depth                 = 6,
        heads                 = 8,
        attn_dropout          = 0.1,
        ff_dropout            = 0.1,
        continuous_mean_std   = torch.stack((scaler["mean"],scaler["std"]),dim=1),
        use_shared_categ_embed = shared_embed
    )
    if torch.cuda.device_count() > 1:
        tt = nn.DataParallel(tt)        # 2GPU 同期データ並列
    tt = tt.to(device)

    # ---------------- 3-6. Optimizer / scheduler -----------------------------
    optimizer = optim.AdamW(tt.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    # ---------------- 3-7. Train loop w/ early-stop --------------------------
    max_epochs      = 200
    patience        = 20
    best_val_rmse   = float("inf")
    best_state      = None
    train_curve, val_curve = [], []

    loss_fn = nn.MSELoss()

    for epoch in range(1,max_epochs+1):
        # ---- train ----
        tt.train(); train_loss=0
        for x_cat,x_cont,y in train_loader:
            x_cat,x_cont,y = x_cat.to(device),x_cont.to(device),y.to(device)
            pred = tt(x_cat,x_cont)
            loss = loss_fn(pred,y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()*len(y)
        train_rmse = math.sqrt(train_loss/len(train_ds))
        train_curve.append(train_rmse)

        # ---- valid ----
        tt.eval(); val_preds, val_trues = [], []
        with torch.no_grad():
            for x_cat,x_cont,y in valid_loader:
                x_cat,x_cont,y = x_cat.to(device),x_cont.to(device),y.to(device)
                pred = tt(x_cat,x_cont)
                val_preds.append(pred.cpu()); val_trues.append(y.cpu())
        val_preds = torch.cat(val_preds); val_trues = torch.cat(val_trues)
        val_rmse  = rmse(val_preds,val_trues)
        val_curve.append(val_rmse)
        scheduler.step(val_rmse)

        print(f"Epoch {epoch:03d}: train RMSE {train_rmse:.5f} | valid RMSE {val_rmse:.5f}")

        # early-stopping
        if val_rmse < best_val_rmse - 1e-5:
            best_val_rmse = val_rmse
            best_state = tt.state_dict()
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early-stopped at epoch {epoch}. Best epoch = {best_epoch}")
                break

    tt.load_state_dict(best_state)

    # --------------- 3-8. Evaluation function --------------------------------
    def evaluate(loader, name):
        tt.eval(); preds, trues = [], []
        with torch.no_grad():
            for x_cat,x_cont,y in loader:
                x_cat,x_cont,y = x_cat.to(device),x_cont.to(device),y.to(device)
                preds.append(tt(x_cat,x_cont).cpu()); trues.append(y.cpu())
        preds = torch.cat(preds).squeeze(); trues = torch.cat(trues).squeeze()
        mask = ~(torch.isnan(preds) | torch.isnan(trues))
        preds, trues = preds[mask], trues[mask]

        rmse_val = rmse(preds,trues)
        r2 = 1 - torch.sum((trues-preds)**2)/torch.sum((trues-trues.mean())**2)
        # sign accuracy
        sign_true = (trues >= 0).int()
        sign_pred = (preds >= 0).int()
        sign_acc  = (sign_true == sign_pred).float().mean().item()
        cm        = confusion_matrix(sign_true, sign_pred, labels=[0,1])
        print(f"\n{name} RMSE {rmse_val:.6f},  R² {r2:.6f},  符号的中率 {sign_acc:.4f}")
        print("Confusion matrix  (row=true, col=pred)")
        print("          Pred-   Pred+")
        print(f"True-     {cm[0,0]:6}  {cm[0,1]:6}")
        print(f"True+     {cm[1,0]:6}  {cm[1,1]:6}")
        return preds.numpy(), trues.numpy()

    # --------------- 3-9. Run eval & save plots ------------------------------
    import matplotlib.pyplot as plt
    preds_val, y_val = evaluate(valid_loader, "Validation")
    preds_test,y_test = evaluate(test_loader , "Test")

    # 曲線
    plt.figure(); plt.plot(train_curve,label="train"); plt.plot(val_curve,label="valid")
    plt.axvline(best_epoch,ls="--",label=f"best={best_epoch}")
    plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.legend()
    plt.title("TabTransformer Training Curve"); 
    plt.savefig(data_dir/"training_curve_tabtransformer.png",dpi=150); plt.close()

    # 残差プロット
    for arr_pred,arr_true,name in [(preds_val,y_val,"valid"),(preds_test,y_test,"test")]:
        residual = arr_true - arr_pred
        plt.figure(figsize=(8,6))
        plt.scatter(arr_pred, residual, alpha=.3)
        plt.axhline(0,ls="--"); plt.xlabel("Predicted"); plt.ylabel("Residual")
        plt.title(f"Residual Plot ({name})"); plt.tight_layout()
        plt.savefig(data_dir/f"residual_plot_{name}_tabtransformer.png",dpi=150)
        plt.close()

    print("Saved outputs to", data_dir)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

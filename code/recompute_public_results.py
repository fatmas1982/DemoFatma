from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, cohen_kappa_score, mean_absolute_error

EXPECTED_N={"T1":720,"T2":474,"T3":460,"T4":474}
BOOT_REPS=2000
BOOT_SEED=20260908


def _read_public(path: Path, model: str):
    df=pd.read_csv(path)
    assert set(df["model"].unique())=={model}
    return df


def _subset(df, target, condition, pred_col):
    x=df[(df.target==target)&(df.condition==condition)].copy()
    assert len(x)==EXPECTED_N[target]
    assert x.Case_ID.astype(str).nunique()==EXPECTED_N[target]
    assert x[pred_col].notna().all()
    return x


def _cluster_boot_delta(df,a,b,reps=BOOT_REPS,seed=BOOT_SEED):
    pids=sorted(df.Prompt_ID.astype(str).unique())
    groups={p:df[df.Prompt_ID.astype(str)==p] for p in pids}
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(reps):
        samp=rng.choice(pids,size=len(pids),replace=True)
        x=pd.concat([groups[p] for p in samp],ignore_index=True)
        vals.append(f1_score(x.y_true,x[b],average="macro")-f1_score(x.y_true,x[a],average="macro"))
    return np.percentile(vals,[2.5,97.5])


def _mcnemar(y,a,b):
    ca=np.asarray(a)==np.asarray(y); cb=np.asarray(b)==np.asarray(y)
    aonly=int(np.sum(ca & ~cb)); bonly=int(np.sum(~ca & cb)); n=aonly+bonly
    p=1.0 if n==0 else float(binomtest(min(aonly,bonly),n=n,p=0.5).pvalue)
    return aonly,bonly,p


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--m3",type=Path,default=Path("data/oof/PUBLIC_OOF_M3.csv"))
    ap.add_argument("--m4",type=Path,default=Path("data/oof/PUBLIC_OOF_M4.csv"))
    ap.add_argument("--out",type=Path,default=Path("_recomputed"))
    args=ap.parse_args(); args.out.mkdir(exist_ok=True)
    m3all=_read_public(args.m3,"M3"); m4all=_read_public(args.m4,"M4")

    rows=[]; paired=[]; mc=[]
    for target in ["T1","T2","T3","T4"]:
        for model,all_df,pred in [("M3",m3all,"M3_pred"),("M4",m4all,"M4_pred")]:
            for cond in ["A","B","C"]:
                d=_subset(all_df,target,cond,pred)
                r={"model":model,"target":target,"condition":cond,"n":len(d),"macro_f1":f1_score(d.y_true,d[pred],average="macro")}
                if target=="T4":
                    r.update(qwk=cohen_kappa_score(d.y_true,d[pred],weights="quadratic"),mae=mean_absolute_error(d.y_true,d[pred]),within_one=float(np.mean(np.abs(d.y_true-d[pred])<=1)))
                rows.append(r)
        m3=_subset(m3all,target,"C","M3_pred"); m4=_subset(m4all,target,"C","M4_pred")
        q=m3[["Case_ID","Prompt_ID","y_true","M3_pred"]].merge(m4[["Case_ID","M4_pred"]],on="Case_ID",validate="one_to_one")
        lo,hi=_cluster_boot_delta(q,"M3_pred","M4_pred")
        paired.append({"target":target,"delta_M4_minus_M3":f1_score(q.y_true,q.M4_pred,average="macro")-f1_score(q.y_true,q.M3_pred,average="macro"),"ci_low":lo,"ci_high":hi})
        a,b,p=_mcnemar(q.y_true,q.M3_pred,q.M4_pred)
        mc.append({"target":target,"M3_only_correct":a,"M4_only_correct":b,"mcnemar_exact_p":p})
    pd.DataFrame(rows).to_csv(args.out/"public_metrics_recomputed.csv",index=False)
    pd.DataFrame(paired).to_csv(args.out/"M4_vs_M3_recomputed.csv",index=False)
    pd.DataFrame(mc).to_csv(args.out/"M4_vs_M3_mcnemar_exact.csv",index=False)
    print("Wrote",args.out)

if __name__=="__main__": main()

"""
P2_55 Follow-up Analyses
========================

Reproduces two post-hoc analyses from frozen M3 OOF predictions WITHOUT retraining:

1) Task-conditioned authorised-context utility:
   For each source-bearing Task x Target, compare M3 condition C vs B using
   paired Prompt_ID-cluster bootstrap (2,000 reps, seed 20260908).

2) Human-machine uncertainty:
   Compare M3-C predictive uncertainty between Human_Easy and Human_Hard cases
   using normalized predictive entropy, top-2 margin, and max probability,
   with Prompt_ID-cluster bootstrap (2,000 reps, seed 20260908).

Authoritative protocol:
  - Frozen primary seed: 20260908
  - No M3 retraining
  - No case exclusion by result
  - Task-conditioned source analysis is exploratory
  - No multiplicity correction across 16 Task x Target cells
"""

from pathlib import Path
import argparse, ast, json
import numpy as np
import pandas as pd

BOOT_REPS = 2000
BOOT_SEED = 20260908
SOURCE_TASKS = ["summarisation","question_answering","source_grounded_generation","multi_entity_context_tracking"]
TARGET_LABELS = {"T1":[0,1],"T2":[0,1],"T3":[0,1],"T4":[1,2,3]}


def macro_from_cm(cm: np.ndarray) -> float:
    tp=np.diag(cm).astype(float); fp=cm.sum(axis=0)-tp; fn=cm.sum(axis=1)-tp
    den=2*tp+fp+fn
    f=np.divide(2*tp,den,out=np.zeros_like(tp),where=den!=0)
    return float(f.mean())


def task_pair_bootstrap(df, labels, reps=BOOT_REPS, seed=BOOT_SEED):
    labs=list(labels); idx={v:i for i,v in enumerate(labs)}; k=len(labs)
    pids=sorted(df["Prompt_ID"].astype(str).unique())
    cm_b=np.zeros((len(pids),k,k),dtype=int); cm_c=np.zeros((len(pids),k,k),dtype=int)
    for gi,pid in enumerate(pids):
        s=df[df["Prompt_ID"].astype(str)==pid]
        for yt,pb,pc in zip(s["y_true"].astype(int),s["pred_B"].astype(int),s["pred_C"].astype(int)):
            if yt in idx and pb in idx: cm_b[gi,idx[yt],idx[pb]]+=1
            if yt in idx and pc in idx: cm_c[gi,idx[yt],idx[pc]]+=1
    rng=np.random.default_rng(seed); sampled=rng.choice(len(pids),size=(reps,len(pids)),replace=True)
    b=cm_b[sampled].sum(axis=1); c=cm_c[sampled].sum(axis=1)
    def batch_macro(cms):
        tp=np.diagonal(cms,axis1=1,axis2=2).astype(float); fp=cms.sum(axis=1)-tp; fn=cms.sum(axis=2)-tp
        den=2*tp+fp+fn
        f=np.divide(2*tp,den,out=np.zeros_like(tp),where=den!=0)
        return f.mean(axis=1)
    delta=batch_macro(c)-batch_macro(b)
    return tuple(np.percentile(delta,[2.5,97.5]))


def run_task_conditioned_source_utility(results_dir: Path) -> pd.DataFrame:
    rows=[]
    for target in ["T1","T2","T3","T4"]:
        B=pd.read_csv(results_dir/f"OOF_M3_{target}_B_seed20260908.csv")[["Case_ID","Prompt_ID","Task_Type","y_true","M3_pred"]].rename(columns={"M3_pred":"pred_B"})
        C=pd.read_csv(results_dir/f"OOF_M3_{target}_C_seed20260908.csv")[["Case_ID","M3_pred"]].rename(columns={"M3_pred":"pred_C"})
        d=B.merge(C,on="Case_ID",validate="one_to_one")
        for task in SOURCE_TASKS:
            s=d[d["Task_Type"]==task].copy(); labs=TARGET_LABELS[target]; ix={v:i for i,v in enumerate(labs)}
            def cm_for(pred_col):
                cm=np.zeros((len(labs),len(labs)),dtype=int)
                for yt,yp in zip(s["y_true"].astype(int),s[pred_col].astype(int)):
                    if yt in ix and yp in ix: cm[ix[yt],ix[yp]]+=1
                return cm
            f_b=macro_from_cm(cm_for("pred_B")); f_c=macro_from_cm(cm_for("pred_C")); lo,hi=task_pair_bootstrap(s,labs)
            rows.append({"target":target,"task":task,"n_cases":len(s),"n_prompts":s["Prompt_ID"].nunique(),"class_counts":json.dumps({int(k):int(v) for k,v in s["y_true"].value_counts().sort_index().items()}),"macro_f1_B":f_b,"macro_f1_C":f_c,"delta_C_minus_B":f_c-f_b,"ci_low":lo,"ci_high":hi,"interval_pattern":"benefit" if lo>0 else ("harm" if hi<0 else "uncertain"),"analysis_status":"exploratory; no multiplicity correction"})
    return pd.DataFrame(rows)


def parse_probs(x): return np.array(ast.literal_eval(x),dtype=float)


def bootstrap_group_diff(df, value_col, reps=BOOT_REPS, seed=BOOT_SEED):
    pids=sorted(df["Prompt_ID"].astype(str).unique()); stats=[]
    for pid in pids:
        s=df[df["Prompt_ID"].astype(str)==pid]; easy=s[s["Human_Easy"].astype(bool)]; hard=s[~s["Human_Easy"].astype(bool)]
        stats.append([easy[value_col].sum(),len(easy),hard[value_col].sum(),len(hard)])
    arr=np.array(stats,float); rng=np.random.default_rng(seed); samp=rng.choice(len(pids),size=(reps,len(pids)),replace=True); sums=arr[samp].sum(axis=1)
    easy_mean=np.divide(sums[:,0],sums[:,1],out=np.full(reps,np.nan),where=sums[:,1]>0)
    hard_mean=np.divide(sums[:,2],sums[:,3],out=np.full(reps,np.nan),where=sums[:,3]>0)
    return tuple(np.nanpercentile(hard_mean-easy_mean,[2.5,97.5]))


def run_human_machine_uncertainty(results_dir: Path) -> pd.DataFrame:
    rows=[]
    for target in ["T1","T2","T3","T4"]:
        df=pd.read_csv(results_dir/f"OOF_M3_{target}_C_seed20260908.csv").copy(); probs=np.stack(df["M3_probs"].map(parse_probs)); k=probs.shape[1]
        df["normalized_entropy"]=-(probs*np.log(np.clip(probs,1e-12,1.0))).sum(axis=1)/np.log(k)
        sorted_p=np.sort(probs,axis=1); df["top2_margin"]=sorted_p[:,-1]-sorted_p[:,-2]; df["max_probability"]=sorted_p[:,-1]
        for col in ["normalized_entropy","top2_margin","max_probability"]:
            easy=df[df["Human_Easy"].astype(bool)][col]; hard=df[~df["Human_Easy"].astype(bool)][col]; lo,hi=bootstrap_group_diff(df,col)
            rows.append({"target":target,"measure":col,"easy_mean":float(easy.mean()),"hard_mean":float(hard.mean()),"hard_minus_easy":float(hard.mean()-easy.mean()),"ci_low":float(lo),"ci_high":float(hi),"interval_pattern":"higher_on_hard" if lo>0 else ("lower_on_hard" if hi<0 else "uncertain"),"interpretation_note":"predictive uncertainty proxy; probabilities are not claimed to be calibrated"})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--results-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,default=Path("./P2_55_followup_outputs")); args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    source_df=run_task_conditioned_source_utility(args.results_dir); unc_df=run_human_machine_uncertainty(args.results_dir)
    source_df.to_csv(args.output_dir/"P2_55_TASK_CONDITIONED_SOURCE_UTILITY.csv",index=False); unc_df.to_csv(args.output_dir/"P2_55_HUMAN_MACHINE_UNCERTAINTY.csv",index=False)
    print("Saved to",args.output_dir.resolve())

if __name__=="__main__": main()

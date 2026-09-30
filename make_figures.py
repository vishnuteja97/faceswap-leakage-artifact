"""Regenerate numerical research figures from released inputs (Matplotlib optional)."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reproduce_part1 import load_rows,TOOLS
from reproduce_part2 import results
from reproduce_controls import transplant_results
DATA=Path(__file__).resolve().parent/'data'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=DATA.parent.parent/'artifact_figures');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    def save(fig,name):
        fig.tight_layout();fig.savefig(a.output/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    rows=load_rows();fig,axes=plt.subplots(2,4,figsize=(13,6))
    for ax,tool in zip(axes.flat,TOOLS):
        rr=[r for r in rows if r['tool']==tool and r['embedding_model']=='buffalo_l']
        for field,label in [('target','Target'),('nonmember','Non-member'),('donor','Donor')]:
            ax.hist([float(r[field+'_sim_median']) for r in rr],bins=30,density=True,alpha=.45,label=label)
        ax.set_title(tool);ax.set_xlabel('Median gallery cosine')
    axes.flat[-1].axis('off');axes.flat[0].legend(fontsize=7);save(fig,'single_swap_distributions')
    fig,ax=plt.subplots(figsize=(5,4))
    for tool in TOOLS:
        rr=[r for r in rows if r['tool']==tool and r['embedding_model']=='buffalo_l']
        p=np.sort([float(r['target_sim_median']) for r in rr]);n=np.sort([float(r['nonmember_sim_median']) for r in rr]);ts=np.r_[np.unique(np.r_[p,n]),np.inf]
        ax.plot(1-np.searchsorted(n,ts)/len(n),1-np.searchsorted(p,ts)/len(p),label=tool)
    ax.set(xlabel='False-positive rate',ylabel='True-positive rate');ax.legend(fontsize=7);save(fig,'single_swap_roc')
    values=results();pred=json.loads((DATA/'recorded/figure5_monte_carlo.json').read_text())['tools']
    fig,axes=plt.subplots(1,3,figsize=(11,3),sharey=True)
    for ax,(tool,r) in zip(axes,values.items()):
        ax.plot(range(1,26),[p['leak_median'] for p in r['series']],label='Measured gallery median')
        ax.plot(range(1,26),pred[tool]['predicted_mean_cosine_to_source'],'--',label='Recorded model mean/source')
        ax.plot(range(1,26),[p['nonmember_median_pooled'] for p in r['series']],':',label='Measured pooled floor')
        ax.set(title=tool,xlabel='Pass');ax.legend(fontsize=6)
    axes[0].set_ylabel('Cosine similarity (different summaries)');save(fig,'predicted_vs_measured')
    for name,key,ylabel in [('cascade_auc','auc','AUC'),('cascade_tpr','tpr_at_1pct_fpr','TPR at historical quantile threshold'),('cascade_excess','excess_median','Excess gallery median')]:
        fig,ax=plt.subplots(figsize=(5,3))
        for tool,r in values.items():ax.plot(range(1,26),[p[key] for p in r['series']],label=tool)
        if key=='auc':ax.axhline(.5,color='gray',ls=':')
        ax.set(xlabel='Pass',ylabel=ylabel);ax.legend();save(fig,name)
    fig,ax=plt.subplots(figsize=(6,3))
    for tool,r in values.items():
        ax.plot(range(1,21),[r['first_excess_ratio']]+[p['excess_ratio'] for p in r['series'][1:20]],label=tool)
        ax.axhline(r['rho_B'],ls='-.',alpha=.4)
    ax.set(xlabel='Transition ending at pass',ylabel='Excess ratio');ax.legend();save(fig,'cascade_ratios')
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    for ax,tool in zip(axes,values):
        B=np.load(DATA/f'operator_{tool}.npz',allow_pickle=False)['B'];e=np.linalg.eigvals(B.astype(float));theta=np.linspace(0,2*np.pi,300)
        ax.plot(np.cos(theta),np.sin(theta),color='gray',ls=':');ax.scatter(e.real,e.imag,s=3);ax.set(title=tool,xlabel='Real',ylabel='Imaginary',aspect='equal')
    save(fig,'operator_spectra')
    tr=transplant_results();fig,axes=plt.subplots(1,3,figsize=(10,3),sharey=True)
    for ax,tool in zip(axes,values):
        for cond,label in [('full','Full'),('face_transplant','Face-only'),('foreign_face','Context-only')]:
            ax.plot([1,5,10,25],[tr[f'{tool}|{cond}|pass{k}']['auc'] for k in [1,5,10,25]],'o-',label=label)
        ax.axhline(.5,color='gray',ls=':');ax.set(title=tool,xlabel='Pass');ax.legend(fontsize=7)
    axes[0].set_ylabel('AUC');save(fig,'face_context_auc')
    lc=json.loads((DATA/'recorded/learning_curve.json').read_text())['rows'];fig,axes=plt.subplots(1,2,figsize=(8,3));lc=sorted(lc,key=lambda x:x['n_train'])
    for ax,key in zip(axes,['rho_B','mean_r2']):ax.plot([r['n_train'] for r in lc],[r[key] for r in lc],'o-');ax.set(xlabel='Training triplets',ylabel=key,title='Recorded learning-curve run')
    save(fig,'learning_curve_recorded')
    print('Wrote 10 PDF figures to',a.output)

if __name__=='__main__':main()

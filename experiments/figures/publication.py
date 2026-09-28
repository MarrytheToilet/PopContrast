"""Regenerate the single-column tradeoff, model/seed, and framework figures.

No experiment or operating-point selection is performed here. Figure 3 uses
the diagnostic full-catalog records; model/seed figures use frozen validation
choices. The framework retains the original artwork with five precise text overlays.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties, findfont
import numpy as np

from .style import apply, PINK, BLUE, INK, MUTE, BG, GOLD
from .sources import source_path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__import__("os").environ.get("POPCONTRAST_PUBLICATION_DIR", ROOT / "results"))
OUT = HERE/'figures/www2027'
SOURCES = {}
VALUES = {}
OUTPUTS = {}

def read(path):
    actual = source_path(path)
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(actual.read_bytes()).hexdigest()
    return json.loads(actual.read_text())

def save(fig, name):
    OUT.mkdir(exist_ok=True, parents=True)
    # Explicit figure inches are retained so the font-size audit is meaningful.
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    assert bounds.x0 >= -.005 and bounds.y0 >= -.005 and bounds.x1 <= width+.005 and bounds.y1 <= height+.005, (name, bounds)
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.png', dpi=300)
    OUTPUTS[name] = {'inches': fig.get_size_inches().tolist(),
                     'files': {str((OUT/f'{name}.{ext}').relative_to(ROOT)):
                               hashlib.sha256((OUT/f'{name}.{ext}').read_bytes()).hexdigest()
                               for ext in ['pdf', 'png']}}
    plt.close(fig)

def pareto():
    apply(plt, 7.2)
    fig, axes = plt.subplots(1, 3, figsize=(3.335, 1.72))
    fig.subplots_adjust(left=.142, right=.985, bottom=.26, top=.72, wspace=.48)
    trace=[]
    for ax, ds in zip(axes, ['beauty','sports','toys']):
        panel=read(ROOT/f'results/main_panel_{ds}.json')
        base=panel['baseline']
        ax.axvline(base['R10'],color=MUTE,lw=.65,ls=(0,(2,3)),alpha=.7)
        ax.axhline(base['cov10'],color=MUTE,lw=.65,ls=(0,(2,3)),alpha=.7)
        for family, color, marker, ls in [('model_pmi',PINK,'o','-'),('naive',BLUE,'s','--')]:
            keys=sorted(panel[family],key=float)
            rows=[(0.,base)]+[(float(k),panel[family][k]) for k in keys]
            x=[r['R10'] for _,r in rows];y=[r['cov10'] for _,r in rows]
            ax.plot(x,y,color=color,lw=1.25,marker=marker,markersize=3.3,
                    mec='white',mew=.5,linestyle=ls,zorder=3)
            trace.extend(dict(dataset=ds,family=family,beta=b,R10=r['R10'],cov10=r['cov10']) for b,r in rows)
        ax.scatter(base['R10'],base['cov10'],s=44,marker='*',color=GOLD,
                   edgecolor='white',lw=.55,zorder=6)
        ax.set_title(ds.capitalize(),pad=3,fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(2))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(length=2,pad=2,labelsize=6.6)
        ax.margins(x=.15,y=.17)
    axes[0].set_ylabel('Coverage@10',labelpad=3)
    fig.supxlabel('Overall R@10',y=.025,fontsize=7.2)
    handles=[Line2D([],[],color=PINK,marker='o',lw=1.2,ms=3.5,label='PopContrast'),
             Line2D([],[],color=BLUE,marker='s',ls='--',lw=1.2,ms=3.2,label='Count discount'),
             Line2D([],[],color=GOLD,marker='*',ls='',ms=6,label='Raw')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,1.005),
               ncol=3,frameon=False,fontsize=6.8,handlelength=1.15,
               handletextpad=.35,columnspacing=.65)
    VALUES['pareto_single_column']=trace
    save(fig,'pareto_single_column')

def model_seed_evidence():
    apply(plt,8.3)
    rows=read(ROOT/'results/benchmark/completed_results.json')
    idx={(r['run'],r['rule']):r for r in rows}
    llm=[('Qwen2.5-1.5B / Beauty','lcrec_qwen1p5b_beauty_s1_compact/evaluation'),
         ('Qwen2.5-3B / Beauty','lcrec_qwen3b_beauty_s1_compact/evaluation_fp32'),
         ('SmolLM2-360M / Clothing','lcrec_smollm360m_clothing_s1/evaluation_fp32'),
         ('SmolLM2-360M / Games','lcrec_smollm360m_games2023_s1/evaluation')]
    fig=plt.figure(figsize=(7.0,2.18))
    gs=fig.add_gridspec(1,3,left=.21,right=.984,bottom=.245,top=.79,
                        width_ratios=[1.35,1,1],wspace=.48)
    ax=fig.add_subplot(gs[0]);traces=[]
    for y,(label,run) in enumerate(llm):
        raw=idx[run,'raw'];r=idx[run,'budget_5pct']
        lo,hi=np.array(r['tail_ci95'])*100;delta=r['tail_delta']*100
        ax.errorbar(delta,y,xerr=[[delta-lo],[hi-delta]],fmt='o',color=PINK,
                    ecolor=PINK,lw=1.4,capsize=2.5,ms=5,mec='white',mew=.6,zorder=3)
        traces.append(dict(run=run,method=r['method'],raw=raw['tailR10'],corrected=r['tailR10'],
                           tail_delta=r['tail_delta'],tail_ci95=r['tail_ci95']))
    ax.axvline(0,color=MUTE,ls=(0,(3,3)),lw=.8)
    ax.set_yticks(range(4),[x[0] for x in llm],fontsize=7.4)
    ax.set_ylim(3.6,-.55);ax.set_xlim(-.4,1.4)
    ax.set_xticks([0,.5,1.0]);ax.set_xlabel(r'$\Delta$Tail R@10 (pp)')
    ax.set_title('(a) LLMs: paired 95% CI',loc='left',fontsize=8.6,pad=10)
    ax.grid(axis='y',visible=False)
    for col,ds in enumerate(['beauty','clothing'],1):
        ax=fig.add_subplot(gs[col]);raws=[];corrected=[]
        for seed in range(3):
            run=f'{ds}_s{seed}_l3'+('_fast' if seed==1 else '')+'/evaluation'
            raw=idx[run,'raw'];r=idx[run,'budget_5pct']
            raws.append(raw['tailR10']*100);corrected.append(r['tailR10']*100)
            traces.append(dict(run=run,method=r['method'],raw=raw['tailR10'],corrected=r['tailR10']))
        x=np.arange(3)
        ax.vlines(x,raws,corrected,color=MUTE,lw=1.6,zorder=2)
        ax.scatter(x,raws,color=BLUE,s=33,marker='s',edgecolor='white',lw=.6,zorder=3)
        ax.scatter(x,corrected,color=PINK,s=35,marker='o',edgecolor='white',lw=.6,zorder=4)
        ax.set_xticks(x,['0','1','2']);ax.set_xlabel('Training seed')
        ax.set_ylabel('Tail R@10 (%)',labelpad=3)
        ax.set_title(f'({chr(97+col)}) TIGER / {ds.capitalize()}',fontsize=8.6,loc='left',pad=10)
        ax.set_xlim(-.4,2.4);ax.margins(y=.25);ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis='x',visible=False)
    handles=[Line2D([],[],marker='s',ls='',ms=5,color=BLUE,label='Raw'),
             Line2D([],[],marker='o',ls='',ms=5,color=PINK,label='PopContrast')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.727,1.02),
               ncol=2,frameon=False,fontsize=8,handlelength=.8,columnspacing=1.3)
    VALUES['model_seed_evidence']=traces
    save(fig,'model_seed_evidence')

def framework():
    """Keep every original icon and formula; overlay only five text regions."""
    path=ROOT/'assets/framework.png'
    SOURCES['assets/framework.png']=hashlib.sha256(path.read_bytes()).hexdigest()
    im=plt.imread(path)
    apply(plt,11)
    fig=plt.figure(figsize=(12,12*957/2048),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.imshow(im,extent=(0,2048,957,0))
    ax.set_xlim(0,2048);ax.set_ylim(957,0);ax.axis('off')
    bundled_font=ROOT/'assets/fonts/ComicNeue-Regular.otf'
    font=FontProperties(fname=str(bundled_font) if bundled_font.is_file() else findfont('Comic Neue'))
    changes=[]
    def replace(rect,text,xy,fontsize,background='white',color='black',weight='normal',align='center'):
        x,y,w,h=rect
        ax.add_patch(Rectangle((x,y),w,h,facecolor=background,edgecolor='none',zorder=4))
        ax.text(*xy,text,ha=align,va='center',fontsize=fontsize,fontproperties=font,
                color=color,fontweight=weight,linespacing=1.05,zorder=5)
        changes.append(dict(rect=rect,text=text,position=xy))
    replace((744,15,550,44),'Diagnosing Popularity Bias',(1019,35),13.1,color='#005ABB',weight='bold')
    replace((730,269,305,44),'Raw beam search',(737,291),12.0,weight='bold',align='left')
    replace((730,480,330,96),'Weak code-density link\nwith item popularity\n'+r'($\rho\approx0.05$--$0.11$)',
            (737,530),10.3,align='left')
    replace((690,903,607,34),r'Full score centering at $\beta=\sigma_m\approx2.4$',
            (993,922),11.7,background='#F4F8FD')
    replace((1564,243,181,34),'Frozen model',(1652,260),11.5,background='#F9E1EC',color='#E95E9E')
    VALUES['framework_refined']={'type':'schematic','source':'assets/framework.png',
                                 'text_overlays':changes,'illustrated_backbone':'TIGER'}
    save(fig,'framework_refined')

def main():
    pareto();model_seed_evidence();framework()
    manifest={'sources':SOURCES,'figures':VALUES,'outputs':OUTPUTS,'policy':'Frozen existing measurements; no test-based selection or smoothing.',
              'style':{'pink':PINK,'blue':BLUE,'font':'DejaVu Sans','framework_font':'Comic Neue'}}
    (HERE/'figure_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Rebuilt single-column Figure 3, LLM/seed evidence and framework text overlays.')

if __name__=='__main__':main()

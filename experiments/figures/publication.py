"""Regenerate publication figures from archived measurements.

No experiment or operating-point selection is performed here. Figure 3 uses
the diagnostic full-catalog records; model/seed figures use frozen validation
choices. Figure identities and panel ordering are described in manuscript
captions and short in-figure identity labels; there are no overall titles.
"""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.transforms import Bbox
import numpy as np

from .style import (apply, apply_reference, PINK, BLUE, INK, MUTE, BG, GOLD,
                    COLUMN_WIDTH, TEXT_WIDTH, REFERENCE_LABEL, REFERENCE_TICK,
                    REFERENCE_LEGEND)
from .sources import source_path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__import__("os").environ.get("POPCONTRAST_PUBLICATION_DIR", ROOT / "results"))
OUT = HERE/'figures'
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
    assert not getattr(fig,'_suptitle',None)
    assert all(not ax.get_title(loc) for ax in fig.axes for loc in ['left','center','right']),name
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    native=getattr(fig,'_popcontrast_original_style',False)
    fixed=getattr(fig,'_popcontrast_print_layout',False)
    if not native or fixed:
        assert bounds.x0 >= -.005 and bounds.y0 >= -.005 and bounds.x1 <= width+.005 and bounds.y1 <= height+.005, (name, bounds)
    renderer = fig.canvas.get_renderer()
    # Locators can create ticks outside the view interval; Matplotlib does not
    # draw those labels, so they are not part of the clipping audit.
    inactive_ticks=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            low,high=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not low-1e-12<=tick.get_loc()<=high+1e-12:
                    inactive_ticks.update([tick.label1,tick.label2])
    for label in fig.findobj(matplotlib.text.Text):
        if label in inactive_ticks or not label.get_visible() or not label.get_text():
            continue
        box = label.get_window_extent(renderer)
        if native and not fixed:
            bounds=Bbox.union([bounds,box.transformed(fig.dpi_scale_trans.inverted())])
        else:
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1+1 and box.y1 <= fig.bbox.y1+1, (name,label.get_text())
    kwargs={'bbox_inches':bounds.padded(.1),'pad_inches':0} if native and not fixed else {}
    fig.savefig(OUT/f'{name}.pdf', metadata={'CreationDate': None},**kwargs)
    fig.savefig(OUT/f'{name}.png', dpi=300,**kwargs)
    if native and not fixed:width,height=bounds.width+.2,bounds.height+.2
    print_width=COLUMN_WIDTH
    panel_inches=[[ax.bbox.width/fig.dpi,ax.bbox.height/fig.dpi] for ax in fig.axes]
    visible_fonts=[label.get_fontsize()*print_width/width
                   for label in fig.findobj(matplotlib.text.Text)
                   if label not in inactive_ticks and label.get_visible() and label.get_text()]
    OUTPUTS[name] = {'inches': [width,height],
                     'renderer':'original make_figures.py' if native else 'publication',
                     'original_builder':getattr(fig,'_popcontrast_original_builder',None),
                     'print_inches':[print_width,height*print_width/width],
                     'panel_inches':panel_inches,
                     'minimum_vector_font_pt':min(visible_fonts) if visible_fonts else None,
                     'axes':len(fig.axes),
                     'titles':[],
                     'identity_labels':getattr(fig,'_popcontrast_identity_labels',[]),
                     'files': {str((OUT/f'{name}.{ext}').relative_to(ROOT)):
                               hashlib.sha256((OUT/f'{name}.{ext}').read_bytes()).hexdigest()
                               for ext in ['pdf', 'png']}}
    plt.close(fig)

def panel_labels(fig, axes, labels, gap=.018):
    """Short dataset identifiers above panels, separate from plot titles."""
    width,height=fig.get_size_inches()
    scale=width/COLUMN_WIDTH
    for ax,label in zip(axes,labels):
        pos=ax.get_position()
        fig.text((pos.x0+pos.x1)/2,pos.y1+gap*scale/height,label,
                 ha='center',va='bottom',fontsize=REFERENCE_TICK*scale,color=INK)
    fig._popcontrast_identity_labels=list(labels)

def marginal():
    from scipy.stats import spearmanr
    fd=archived_plot_data();trace=[]
    for ds,d in fd.items():
        lp=d['log_pop'];mg=d['marginal']
        trace.append(dict(dataset=ds,n_items=len(lp),log_pop_sha256=array_hash(lp),
                          marginal_sha256=array_hash(mg),rho=float(spearmanr(mg,lp).statistic),
                          trend_coefficients=np.polyfit(lp,mg,1).tolist()))
    VALUES['marginal_single_column']=trace
    save(native_figure('fig4_marginal_pop',fd),'marginal_single_column')

def rankshift():
    path=source_path(ROOT/'assets/plot_data/rankshift_beauty.npz')
    SOURCES['assets/plot_data/rankshift_beauty.npz']=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as data:lp=data['logpop'];dr=data['drank']
    bins=np.quantile(lp,np.linspace(0,1,13));mids=[];meds=[]
    for a,b in zip(bins[:-1],bins[1:]):
        mask=(lp>=a)&(lp<=b)
        if mask.sum()>20:mids.append(float((a+b)/2));meds.append(float(np.median(dr[mask])))
    VALUES['rankshift_single_column']=dict(dataset='beauty',beta=.75,n_items=len(lp),
        log_pop_sha256=array_hash(lp),rank_change_sha256=array_hash(dr),
        bin_edges=bins.tolist(),median_x=mids,median_y=meds)
    save(native_figure('fig8_rankshift',path),'rankshift_single_column')

def pareto():
    apply_reference(plt)
    fig, axes = plt.subplots(1, 3, figsize=(COLUMN_WIDTH, 1.35))
    place_panels(fig,axes,left=.47,right=.04,gap=.30,bottom=.32,top=.36,square=True)
    trace=[]
    for ax, ds in zip(axes, ['beauty','sports','toys']):
        panel=read(ROOT/f'results/main_panel_{ds}.json')
        base=panel['baseline']
        ax.axvline(base['R10'],color=MUTE,lw=.65,ls=(0,(2,3)),alpha=.7)
        ax.axhline(base['cov10'],color=MUTE,lw=.65,ls=(0,(2,3)),alpha=.7)
        for family, color, marker, ls in [('model_pmi',PINK,'o','-'),('naive',BLUE,'o','-')]:
            keys=sorted(panel[family],key=float)
            rows=[(0.,base)]+[(float(k),panel[family][k]) for k in keys]
            x=[r['R10'] for _,r in rows];y=[r['cov10'] for _,r in rows]
            ax.plot(x,y,color=color,lw=.9,marker=marker,markersize=1.9,
                    mec=BG,mew=.3,linestyle=ls,zorder=3)
            trace.extend(dict(dataset=ds,family=family,beta=b,R10=r['R10'],cov10=r['cov10']) for b,r in rows)
        ax.scatter(base['R10'],base['cov10'],s=18,marker='*',color=GOLD,
                   edgecolor=BG,lw=.35,zorder=6)
        ax.xaxis.set_major_locator(MaxNLocator(2))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(length=2,pad=2)
        ax.margins(x=.15,y=.14)
    axes[0].set_ylabel('Coverage@10',labelpad=3)
    fig.supxlabel('Overall R@10',y=.025,fontsize=REFERENCE_LABEL)
    handles=[Line2D([],[],color=PINK,lw=.9,label='PopContrast'),
             Line2D([],[],color=BLUE,lw=.9,label='Count discount'),
             Line2D([],[],color=GOLD,marker='*',ls='',ms=4,label='Raw')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.99),
               ncol=3,borderaxespad=0,borderpad=0,handlelength=1.15,
               handletextpad=.35,columnspacing=.75)
    panel_labels(fig,axes,['Beauty','Sports','Toys'],gap=.035)
    VALUES['pareto_single_column']=trace
    save(fig,'pareto_single_column')

def model_seed_evidence():
    """Bars encode every checkpoint, with paired user intervals on LLM gains."""
    apply_reference(plt)
    rows=read(ROOT/'results/benchmark/completed_results.json')
    idx={(r['run'],r['rule']):r for r in rows}
    llm=[('Q1','lcrec_qwen1p5b_beauty_s1_compact/evaluation'),
         ('Q3','lcrec_qwen3b_beauty_s1_compact/evaluation_fp32'),
         ('SC','lcrec_smollm360m_clothing_s1/evaluation_fp32'),
         ('SG','lcrec_smollm360m_games2023_s1/evaluation')]
    fig,axes=plt.subplots(1,3,figsize=(COLUMN_WIDTH,1.35))
    place_panels(fig,axes,left=.45,right=.03,gap=[.42,.12],bottom=.30,top=.36,square=True)
    ax=axes[0];traces=[]
    for y,(label,run) in enumerate(llm):
        raw=idx[run,'raw'];r=idx[run,'budget_5pct']
        lo,hi=np.array(r['tail_ci95'])*100;delta=r['tail_delta']*100
        ax.bar(y,delta,width=.60,color=PINK,edgecolor=BG,linewidth=.3,
               yerr=np.array([[delta-lo],[hi-delta]]),
               error_kw={'ecolor':INK,'elinewidth':.45,'capsize':1.0,'capthick':.45},zorder=3)
        traces.append(dict(run=run,method=r['method'],raw=raw['tailR10'],corrected=r['tailR10'],
                           tail_delta=r['tail_delta'],tail_ci95=r['tail_ci95']))
    ax.axhline(0,color=MUTE,lw=.55)
    ax.set_xticks(range(4),[x[0] for x in llm])
    ax.set_xlim(-.6,3.6);ax.set_ylim(-.35,1.4)
    ax.set_yticks([0,.5,1]);ax.set_ylabel(r'$\Delta$Tail R@10 (pp)',labelpad=2)
    ax.grid(axis='x',visible=False)
    for row,ds in enumerate(['beauty','clothing'],1):
        ax=axes[row];raws=[];corrected=[]
        for seed in range(3):
            run=f'{ds}_s{seed}_l3'+('_fast' if seed==1 else '')+'/evaluation'
            raw=idx[run,'raw'];r=idx[run,'budget_5pct']
            raws.append(raw['tailR10']*100);corrected.append(r['tailR10']*100)
            traces.append(dict(run=run,method=r['method'],raw=raw['tailR10'],corrected=r['tailR10']))
        x=np.arange(3)
        width=.30
        ax.bar(x-width/2,raws,width,color=BLUE,edgecolor=BG,linewidth=.4,label='Raw',zorder=3)
        ax.bar(x+width/2,corrected,width,color=PINK,edgecolor=BG,linewidth=.4,label='PopContrast',zorder=3)
        ax.set_xticks(x,['s0','s1','s2'])
        if row==1:ax.set_ylabel('Tail R@10 (%)',labelpad=2)
        ax.set_xlim(-.55,2.55);ax.set_ylim(0,1.75)
        ax.set_yticks([0,.5,1,1.5]);ax.grid(axis='x',visible=False)
        if row==2:ax.tick_params(axis='y',labelleft=False)
    handles=[Rectangle((0,0),1,1,facecolor=BLUE,label='Raw'),
             Rectangle((0,0),1,1,facecolor=PINK,label='PopContrast')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.61,.99),
               ncol=2,borderaxespad=0,borderpad=0,handlelength=1.0)
    panel_labels(fig,axes,['(a) LLMs','(b) Beauty','(c) Clothing'],gap=.035)
    fig.text(.52,.012,'Q1/Q3: Beauty; SC: Clothing; SG: Games',
             ha='center',va='bottom',fontsize=REFERENCE_TICK,color=INK)
    VALUES['model_seed_evidence']=traces
    save(fig,'model_seed_evidence')
    OUTPUTS['model_seed_evidence'].update(chart='bars',seed_ylim=[0,1.75],
                                         llm_intervals='paired 95% user-bootstrap')


def archived_plot_data():
    fd={}
    for ds in ['beauty','sports','toys']:
        relative=f'assets/plot_data/figdata_{ds}.npz';path=source_path(ROOT/relative)
        SOURCES[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as data:fd[ds]={key:data[key].copy() for key in data.files}
    return fd


def native_figure(function,*args):
    """Reuse the renderer that created the author's preferred original plots."""
    from experiments import make_figures as original
    with plt.rc_context(original.STYLE):
        fig=getattr(original,function)(*args,return_figure=True)
    assert fig is not None,function
    fig._popcontrast_original_style=True
    fig._popcontrast_original_builder=function
    if function=='fig9b_quintile_heatmap':
        # Restore the author's compact heatmap, with explicit leading zeros.
        for ax in fig.axes:
            ax.set_xticklabels(['0.25','0.5','0.75','1'])
            pos=ax.get_position()
            ax.set_position([pos.x0,pos.y0,pos.width,pos.height-.22/fig.get_figheight()])
        panel_labels(fig,fig.axes,['Beauty','Sports','Toys'])
    else:format_native_print(fig,function)
    return fig


def place_panels(fig,axes,left,right,gap,bottom,top,square=False):
    """Lay out plots in physical inches, including space for readable ticks."""
    width,height=fig.get_size_inches()
    gaps=[gap]*(len(axes)-1) if np.isscalar(gap) else gap
    panel_width=(width-left-right-sum(gaps))/len(axes)
    panel_height=panel_width if square else height-bottom-top
    x=left
    for i,ax in enumerate(axes):
        ax.set_position([x/width,bottom/height,panel_width/width,panel_height/height])
        if square:ax.set_box_aspect(1)
        x+=panel_width+(gaps[i] if i<len(gaps) else 0)


def format_native_print(fig,builder):
    """Preserve measured curves and fills; format typography for print."""
    original_width=fig.get_size_inches()[0]
    mark_scale=COLUMN_WIDTH/original_width
    apply_reference(plt)
    fig._popcontrast_print_layout=True
    # Preserve the visual weight of the original curves and density cells.
    for ax in fig.axes:
        for line in ax.lines:
            line.set_linewidth(line.get_linewidth()*mark_scale)
            line.set_markersize(line.get_markersize()*mark_scale)
            line.set_markeredgewidth(line.get_markeredgewidth()*mark_scale)
        for collection in ax.collections:
            collection.set_linewidth(np.asarray(collection.get_linewidths())*mark_scale)
        for spine in ax.spines.values():spine.set_linewidth(.4)
        ax.tick_params(labelsize=REFERENCE_TICK,length=2,pad=2,width=.4)
        for line in ax.get_xgridlines()+ax.get_ygridlines():line.set_linewidth(.33)
        for label in [ax.xaxis.label,ax.yaxis.label]:label.set_fontsize(REFERENCE_LABEL)
        for label in ax.texts:label.set_fontsize(7)
        legend=ax.get_legend()
        if legend:
            for label in legend.get_texts():label.set_fontsize(REFERENCE_LEGEND)
    axes=fig.axes
    if builder=='fig8_rankshift':
        fig.set_size_inches(COLUMN_WIDTH,2.10)
        place_panels(fig,axes,left=.49,right=.06,gap=0,bottom=.35,top=.08)
        for label in axes[0].texts:label.set_fontsize(8)
        axes[0].xaxis.set_major_locator(MaxNLocator(4))
        axes[0].yaxis.set_major_locator(MaxNLocator(4))
        # Rebuild after resizing so legend padding is also in print-size points.
        axes[0].legend(loc='lower right',frameon=True,framealpha=.85,
                       edgecolor='none',fontsize=7,handlelength=1.3,borderpad=.3)
        axes[0].text(.03,.97,'Beauty',transform=axes[0].transAxes,
                     ha='left',va='top',fontsize=REFERENCE_TICK,color=INK)
        fig._popcontrast_identity_labels=['Beauty']
        return
    for ax in axes:ax.set_xlabel('')
    if builder=='fig4_marginal_pop':
        fig.set_size_inches(COLUMN_WIDTH,1.32)
        place_panels(fig,axes,left=.45,right=.035,gap=.25,bottom=.32,top=.16)
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.texts[0].set_position((.04,.95))
            ax.texts[0].set_verticalalignment('top')
        fig.supxlabel('log item popularity',y=.025,fontsize=REFERENCE_LABEL)
    elif builder=='fig10_exposure_stream':
        fig.set_size_inches(COLUMN_WIDTH,1.18)
        place_panels(fig,axes,left=.43,right=.065,gap=.12,bottom=.30,top=.16)
        for ax in axes:
            ax.set_yticks([0,.5,1])
            for label in ax.texts:label.set_fontsize(6.8)
        axes[0].set_ylabel('exposure share')
        # Beauty's two narrowest bands need horizontal separation at this height.
        entry=VALUES['exposure_single_column'][0]
        xs=np.asarray(entry['betas']);shares=np.asarray(entry['shares'])
        x=.68*xs[-1]
        y=1-np.interp(x,xs,shares[:,0])-np.interp(x,xs,shares[:,1])/2
        next(label for label in axes[0].texts if label.get_text()=='q2').set_position((x,y))
        fig.supxlabel(r'$\beta$',y=.025,fontsize=REFERENCE_LABEL)
    elif builder=='fig3_lorenz':
        fig.set_size_inches(COLUMN_WIDTH,1.38)
        place_panels(fig,axes,left=.45,right=.09,gap=.14,bottom=.32,top=.30)
        for i,ax in enumerate(axes):
            ax.set_ylabel('')
            ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
            if i:ax.tick_params(axis='y',labelleft=False)
            if ax.get_legend():ax.get_legend().remove()
        axes[0].set_ylabel('cumulative exposure',labelpad=3)
        fig.supxlabel('items (least → most exposed)',y=.025,fontsize=REFERENCE_LABEL)
        handles=[Line2D([],[],color=BLUE,lw=.8,label='Raw'),
                 Line2D([],[],color=PINK,lw=.8,label=r'PopContrast ($\beta=1$)')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.99),
                   ncol=2,borderaxespad=0,borderpad=0,fontsize=REFERENCE_LEGEND,handlelength=1.3)
    elif builder=='fig2_diversification':
        primary=[ax for ax in axes if ax.get_xlabel()=='' and len(ax.collections)]
        secondary=[ax for ax in axes if ax not in primary]
        assert len(primary)==len(secondary)==3
        # Match the adjacent Lorenz figure: broad, tightly spaced panels,
        # quiet outer ticks, thin curves, and one compact legend.
        fig.set_size_inches(COLUMN_WIDTH,1.38)
        place_panels(fig,primary,left=.33,right=.35,gap=.14,bottom=.32,top=.30)
        for i,(ax,other) in enumerate(zip(primary,secondary)):
            other.set_position(ax.get_position())
            for a in [ax,other]:
                a.set_ylabel('')
                a.tick_params(axis='y',colors=INK)
                for line in a.lines:
                    line.set_marker('')
                    line.set_linewidth(.65)
            ax.set_ylim(.08,.40)
            ax.set_yticks([.1,.2,.3,.4])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.1f}'))
            ax.tick_params(axis='y',labelleft=(i==0))
            other.set_ylim(.62,.84)
            other.set_yticks([.64,.72,.80])
            other.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.2f}'))
            other.tick_params(axis='y',right=(i==2),labelright=(i==2))
            other.spines['right'].set_visible(i==2)
            for line in other.lines:line.set_linestyle((0,(3,2)))
            ax.set_xlim(0,2)
            ax.set_xticks([0,1,2])
        fig.supxlabel(r'$\beta$',y=.025,fontsize=REFERENCE_LABEL)
        handles=[Line2D([],[],color=PINK,lw=.8,label='Coverage@10'),
                 Line2D([],[],color=BLUE,lw=.8,ls=(0,(3,2)),label='Norm. entropy')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.50,.99),
                   ncol=2,borderaxespad=0,borderpad=0,fontsize=REFERENCE_LEGEND,handlelength=1.3)
    else:raise ValueError(builder)
    label_axes=primary if builder=='fig2_diversification' else axes
    panel_labels(fig,label_axes,['Beauty','Sports','Toys'],
                 gap=.018 if builder=='fig10_exposure_stream' else .035)

def quintile():
    data=read(ROOT/'results/enrich_analysis.json');betas=['0.25','0.5','0.75','1.0'];traces=[]
    for ds in ['beauty','sports','toys']:
        rows=data[ds]['quintile_recall']
        base=np.array(rows['0.0']['by_quintile'],dtype=float)
        current=np.array([rows[b]['by_quintile'] for b in betas],dtype=float).T
        gains=np.full_like(current,np.nan);nonzero=base>0
        gains[nonzero]=100*(current[nonzero]-base[nonzero,None])/base[nonzero,None]
        traces.append(dict(dataset=ds,betas=[float(b) for b in betas],baseline=base.tolist(),
            recall=current.tolist(),relative_percent=[[None if np.isnan(v) else float(v) for v in row] for row in gains],
            from_zero=((base[:,None]==0)&(current>0)).tolist()))
    VALUES['quintile_single_column']=traces
    save(native_figure('fig9b_quintile_heatmap',source_path(ROOT/'results/enrich_analysis.json')),
         'quintile_single_column')
    OUTPUTS['quintile_single_column']['typography']='original compact heatmap restored at author request'

def exposure():
    data=read(ROOT/'results/exposure_shares.json');traces=[]
    for ds in ['beauty','sports','toys']:
        beta=data[ds]['betas'];shares=np.array(data[ds]['shares'])
        assert shares.shape==(len(beta),5) and np.allclose(shares.sum(axis=1),1)
        traces.append(dict(dataset=ds,betas=beta,shares=shares.tolist()))
    VALUES['exposure_single_column']=traces
    save(native_figure('fig10_exposure_stream',source_path(ROOT/'results/exposure_shares.json')),
         'exposure_single_column')

def array_hash(array):
    return hashlib.sha256(np.asarray(array,dtype='<f8').tobytes()).hexdigest()


def lorenz():
    fd=archived_plot_data();traces=[]
    for ds,d in fd.items():
        lines=[]
        for key in ['cnt_base','cnt_beta']:
            counts=np.asarray(d[key],dtype=float)
            x=np.r_[0,np.arange(1,len(counts)+1)/len(counts)]
            y=np.r_[0,np.cumsum(np.sort(counts))/counts.sum()]
            lines.append(dict(family=key,n_items=len(counts),total_exposure=float(counts.sum()),
                              x_sha256=array_hash(x),y_sha256=array_hash(y)))
        traces.append(dict(dataset=ds,beta=float(d['beta']),lines=lines))
    VALUES['lorenz_single_column']=traces
    save(native_figure('fig3_lorenz',fd),'lorenz_single_column')

def diversification():
    panels={ds:read(ROOT/f'results/main_panel_{ds}.json') for ds in ['beauty','sports','toys']}
    traces=[]
    for ds,panel in panels.items():
        rows=[(0.,panel['baseline'])]+[(float(b),panel['model_pmi'][b]) for b in sorted(panel['model_pmi'],key=float)]
        traces.append(dict(dataset=ds,betas=[b for b,_ in rows],coverage=[r['cov10'] for _,r in rows],
                           entropy=[r['ent'] for _,r in rows]))
    VALUES['diversification_single_column']=traces
    save(native_figure('fig2_diversification',panels),'diversification_single_column')
    OUTPUTS['diversification_single_column']['style_reference']='lorenz_single_column'
    OUTPUTS['diversification_single_column']['shared_metric_axes']={
        'coverage':[.08,.40],'entropy':[.62,.84]}
    OUTPUTS['diversification_single_column']['markers']='none'

def matched_tradeoffs():
    from .tradeoff_matrix import read_evidence, draw_matrix
    evidence,sources=read_evidence()
    SOURCES.update(sources)
    OUT.mkdir(parents=True,exist_ok=True)
    layout=draw_matrix(evidence,OUT,stem='matched_tradeoffs_compact')
    VALUES['matched_tradeoffs_compact']={'datasets':evidence,'layout':layout}
    OUTPUTS['matched_tradeoffs_compact']={
        'inches':layout['figure_inches'],
        'print_inches':layout['figure_inches'],
        'minimum_vector_font_pt':layout['minimum_vector_font_pt'],
        'titles':[],
        'identity_labels':layout['row_order']+layout['column_order'],
        'files':{str((OUT/f'matched_tradeoffs_compact.{ext}').relative_to(ROOT)):
                 hashlib.sha256((OUT/f'matched_tradeoffs_compact.{ext}').read_bytes()).hexdigest()
                 for ext in ['pdf','png']}}

def popcontrast_overview():
    """Copy the author's vector PDF and preview without altering the artwork."""
    import fitz
    name='popcontrast_overview'
    source='assets/diagrams/popcontrast_overview.pdf'
    preview='assets/diagrams/popcontrast_overview.png'
    path=source_path(ROOT/source)
    preview_path=source_path(ROOT/preview)
    SOURCES[source]=hashlib.sha256(path.read_bytes()).hexdigest()
    SOURCES[preview]=hashlib.sha256(preview_path.read_bytes()).hexdigest()
    pixmap=fitz.Pixmap(str(preview_path))
    width=TEXT_WIDTH
    with fitz.open(path) as doc:
        assert len(doc)==1
        page=doc[0]
        height=width*page.rect.height/page.rect.width
        scale=width*72/page.rect.width
        fonts=[span['size']*scale for block in page.get_text('dict')['blocks']
               if 'lines' in block for line in block['lines']
               for span in line['spans'] if span['text'].strip()]
        native_inches=[page.rect.width/72,page.rect.height/72]
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(preview_path,OUT/f'{name}.png')
    shutil.copyfile(path,OUT/f'{name}.pdf')
    VALUES[name]={'type':'schematic','source':source,'preview':preview,
                  'reference':'geometric','history_average':'mean log-score',
                  'standardization_axis':'catalog items',
                  'candidate_set_preserved':True,
                  'illustrative_rankings':{'before':['A','C','B','D'],
                                           'after':['A','B','C','D'],'K':2},
                  'validation_rule':'budget_5pct',
                  'illustrated_backbone':'generic SID recommender'}
    OUTPUTS[name]={'inches':native_inches,'print_inches':[width,height],
                   'renderer':'author-supplied vector PDF',
                   'raster_pixels':[pixmap.width,pixmap.height],
                   'effective_dpi':pixmap.width/width,
                   'minimum_vector_font_pt':min(fonts),'titles':[],
                   'files':{str((OUT/f'{name}.{ext}').relative_to(ROOT)):
                            hashlib.sha256((OUT/f'{name}.{ext}').read_bytes()).hexdigest()
                            for ext in ['pdf','png']}}

def main():
    marginal();popcontrast_overview();pareto();exposure();quintile();rankshift()
    model_seed_evidence();lorenz();diversification();matched_tradeoffs()
    manifest={'sources':SOURCES,'figures':VALUES,'outputs':OUTPUTS,'policy':'Frozen existing measurements; no test-based selection or smoothing.',
              'style':{'pink':PINK,'blue':BLUE,'font':'DejaVu Sans','framework_font':'author-supplied vector lettering',
                       'reference':'Original Figure 1 palette; enlarged typography at actual print width',
                       'label_pt':REFERENCE_LABEL,'tick_pt':REFERENCE_TICK,'titles':'no overall titles; short identity labels retained'},
              'layout':{'three_panel_figures':'single-column, three horizontal panels',
                        'checkpoint_comparisons':'bars; paired user intervals retained on LLM differences',
                        'titles':'no overall titles; dataset and comparator labels in figures and captions'}}
    (HERE/'figure_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Rebuilt all ten figures with identity labels and no overall titles; source measurements preserved.')

if __name__=='__main__':main()

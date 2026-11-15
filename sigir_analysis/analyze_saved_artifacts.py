"""Recompute SIGIR research diagnostics from immutable saved tables only.

Run: python sigir_analysis/analyze_saved_artifacts.py
No retrieval/model/index libraries are imported, and no input is overwritten.
"""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'sigir_analysis'
sys.path.insert(0, str(OUT / '.plot_deps'))
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplconfig'))
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = ROOT / 'msmarco_large/msmarco_1000000_q1000'
RESEARCH = BASE / 'research_grade_validation'
CTRL = BASE / 'cura_controller'
EFS = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256]
TOL = 0.025
B = 20000
SEED = 20261115
INPUTS = {}
STATS = []


def read(path):
    INPUTS[str(path.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, dtype={'query_key': str})


def ci(x):
    return [float(v) for v in np.quantile(x, [0.025, 0.975])]


def paired_boot(a, b, seed):
    """Positive difference/saving means a costs more than b; pairs stay intact."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), (B, len(a)))
    aa, bb = a[indices].mean(axis=1), b[indices].mean(axis=1)
    return {'difference': float(np.mean(a-b)), 'difference_ci': ci(aa-bb),
            'saving_pct': float(100*(1-b.mean()/a.mean())),
            'saving_pct_ci': ci(100*(1-bb/aa))}


def signflip(delta, seed):
    d = np.asarray(delta, float)
    d = d[np.abs(d) > 1e-12]
    if not len(d):
        return 1., 'all differences zero', 0
    observed = abs(d.sum())
    if len(d) <= 18:
        sums = np.array([0.])
        for value in d:
            sums = np.concatenate([sums+value, sums-value])
        return float(np.mean(np.abs(sums) >= observed-1e-10)), 'exact paired sign-flip', len(d)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(B // 1000):
        signs = rng.choice([-1., 1.], size=(1000, len(d)))
        extreme += int((np.abs(signs @ d) >= observed-1e-10).sum())
    return (extreme+1)/(B+1), 'Monte Carlo paired sign-flip', len(d)


def add_test(cohort, comparison, metric, delta, seed, difference_ci=None):
    values = np.asarray(delta, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (B, len(values)))].mean(axis=1)
    p, mode, nonzero = signflip(values, seed+1)
    lo, hi = ci(means) if difference_ci is None else difference_ci
    STATS.append(dict(cohort=cohort, comparison=comparison, metric=metric,
                      queries=len(values), nonzero_pairs=nonzero, mean_difference=values.mean(),
                      ci95_low=lo, ci95_high=hi, p_value=p, test=mode,
                      interpretation='Exploratory, unadjusted; pairing/exchangeability assumed'))
    nz=values[np.abs(values)>1e-12]
    STATS.append(dict(cohort=cohort,comparison=comparison,metric=metric+' direction',
                      queries=len(values),nonzero_pairs=len(nz),
                      p_value=binomtest(int((nz>0).sum()),len(nz),.5).pvalue if len(nz) else 1.,
                      test='two-sided exact paired sign test, ties excluded',
                      interpretation='Targets direction among non-ties, not the mean effect'))


def wilson(k, n):
    z = 1.96  # matches the saved calibration protocol
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0., center-half), min(1., center+half)


def savefig(fig, name):
    fig.savefig(OUT / name, dpi=190, bbox_inches='tight')
    plt.close(fig)


def metric_row(frame, method, split, effective=False):
    violations = int((frame.nDCG_loss > TOL).sum())
    low, high = wilson(violations, len(frame))
    work = 'effective_distance_computations' if effective else 'distance_computations'
    lat = 'effective_latency_ms' if effective else 'mean_latency_ms'
    return dict(split=split, method=method, query_count=len(frame),
                efSearch=int(frame.efSearch.iloc[0]) if frame.efSearch.nunique()==1 else np.nan,
                mean_efSearch=frame.efSearch.mean(), IR_violations=violations,
                IR_violation_rate=violations/len(frame), wilson95_low=low, wilson95_upper=high,
                ANN_Recall_at_10=frame['ANN_Recall@10'].mean(), mean_nDCG_at_10=frame['nDCG@10'].mean(),
                mean_distance_computations=frame[work].mean(), mean_latency_ms=frame[lat].mean(),
                p95_latency_ms=frame[lat].quantile(.95),
                latency_scope=('sum of saved five-repeat search means and separate probe timing; excludes model/feature overhead' if method=='CURA-Full' else 'saved per-query five-repeat search means'),
                work_scope='probe plus selected search, or probe only at ef10' if method=='CURA-Full' else 'single fixed search')


def md(frame, columns=None):
    f = frame if columns is None else frame[columns]
    # No optional tabulate dependency.
    def val(v):
        if pd.isna(v): return 'NA'
        return f'{v:.6g}' if isinstance(v, (float, np.floating)) else str(v)
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(val(v) for v in row)+' |' for row in f.itertuples(index=False, name=None))


def validate_selection(frame, budget, ids):
    assert frame.query_key.is_unique and set(frame.query_key)==set(ids)
    merged = frame.merge(budget, on=['query_key','efSearch'], suffixes=('_selected','_saved'), validate='one_to_one')
    assert len(merged)==len(ids)
    for col in ['ANN_Recall@10','nDCG@10','nDCG_loss','distance_computations','mean_latency_ms']:
        np.testing.assert_allclose(merged[col+'_selected'], merged[col+'_saved'], atol=1e-12, rtol=1e-10)
    np.testing.assert_array_equal(frame.IR_unsafe, (frame.nDCG_loss>TOL).astype(int))


def exact_split(reference, fixed, cura):
    joined = fixed[['query_key','effective_distance_computations','nDCG@10','IR_unsafe']].merge(
        cura[['query_key','effective_distance_computations','nDCG@10','IR_unsafe']],
        on='query_key', suffixes=('_fixed','_cura'), validate='one_to_one').merge(
        reference[['query_key','exact_nDCG@10']], on='query_key', validate='one_to_one')
    joined['exact_group'] = np.where(joined['exact_nDCG@10']==0, 'exact_zero', 'exact_positive')
    joined.to_csv(OUT/'exact_ndcg_stratified_per_query.csv', index=False)
    rows, savings = [], {}
    for i, (group, g) in enumerate(joined.groupby('exact_group', sort=True)):
        s = paired_boot(g.effective_distance_computations_fixed, g.effective_distance_computations_cura, SEED+i)
        savings[group] = s
        for name, f in [('Fixed-IR',fixed),('CURA-Full',cura)]:
            f = f[f.query_key.isin(g.query_key)]
            row = metric_row(f,name,'MSMARCO_TEST',True)
            row.update(exact_group=group, group_percentage=100*len(g)/len(joined),
                       fixed_minus_cura_distance_computations=s['difference'],
                       work_difference_ci95_low=s['difference_ci'][0], work_difference_ci95_high=s['difference_ci'][1],
                       CURA_work_saving_vs_fixed_pct=s['saving_pct'],
                       saving_ci95_low_pct=s['saving_pct_ci'][0], saving_ci95_high_pct=s['saving_pct_ci'][1],
                       contribution_to_total_work_reduction_pct=100*(g.effective_distance_computations_fixed-g.effective_distance_computations_cura).sum()/(joined.effective_distance_computations_fixed-joined.effective_distance_computations_cura).sum())
            rows.append(row)
        add_test(group,'Fixed-IR minus CURA-Full','effective distance computations',g.effective_distance_computations_fixed-g.effective_distance_computations_cura,SEED+10+i,s['difference_ci'])
        add_test(group,'Fixed-IR minus CURA-Full','nDCG@10',g['nDCG@10_fixed']-g['nDCG@10_cura'],SEED+20+i)
        add_test(group,'CURA-Full minus Fixed-IR','IR violation indicator',g.IR_unsafe_cura-g.IR_unsafe_fixed,SEED+30+i)
        b = int(((g.IR_unsafe_fixed==0)&(g.IR_unsafe_cura==1)).sum())
        c = int(((g.IR_unsafe_fixed==1)&(g.IR_unsafe_cura==0)).sum())
        STATS.append(dict(cohort=group,comparison='CURA-Full vs Fixed-IR',metric='IR violation discordances',queries=len(g),nonzero_pairs=b+c,p_value=binomtest(b,b+c,.5).pvalue if b+c else 1.,test='exact McNemar',cura_only_failures=b,fixed_only_failures=c,interpretation='Few discordances; no equivalence or risk guarantee'))
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'exact_ndcg_stratified_results.csv',index=False)
    fig, ax = plt.subplots(figsize=(8,4.8))
    groups=['exact_zero','exact_positive']; xx=np.arange(2)
    for j,name in enumerate(['Fixed-IR','CURA-Full']):
        y=[result[(result.exact_group==g)&(result.method==name)].mean_distance_computations.iloc[0] for g in groups]
        bars=ax.bar(xx+(j-.5)*.32,y,.32,label=name,color=['#526579','#138d82'][j])
        ax.bar_label(bars,fmt='%.0f',padding=3)
    labels=[]
    for g in groups:
        row=result[(result.exact_group==g)&(result.method=='CURA-Full')].iloc[0]
        labels.append(f"{'Exact nDCG = 0' if g=='exact_zero' else 'Exact nDCG > 0'}\nn={row.query_count}; saving {row.CURA_work_saving_vs_fixed_pct:.1f}%\n95% CI [{row.saving_ci95_low_pct:.1f}, {row.saving_ci95_high_pct:.1f}]%")
    ax.set_xticks(xx,labels);ax.set_ylabel('Mean distance computations (probe included)')
    ax.set_title('Held-out MS MARCO: savings remain in exact-positive queries\nVersus calibration-selected fixed ef96; not risk-matched',fontsize=11)
    ax.legend();ax.set_ylim(0,result.mean_distance_computations.max()*1.2)
    savefig(fig,'exact_ndcg_stratified_work.png')
    positive=result[(result.exact_group=='exact_positive')&(result.method=='CURA-Full')].iloc[0]
    text='# Exact nDCG zero versus positive\n\n'+md(result,['exact_group','method','query_count','group_percentage','mean_efSearch','IR_violation_rate','ANN_Recall_at_10','mean_nDCG_at_10','mean_distance_computations','mean_latency_ms','p95_latency_ms','CURA_work_saving_vs_fixed_pct','saving_ci95_low_pct','saving_ci95_high_pct'])
    text+=f'\n\nSavings are not concentrated in exact-zero queries: exact-positive queries contribute {positive.contribution_to_total_work_reduction_pct:.2f}% of the total fixed-minus-CURA distance reduction. This addresses the narrow zero-reference explanation, not the broader sparse-judgment confound. Exact-zero queries are automatically safe under the nonnegative one-sided loss max(0, exact nDCG minus approximate nDCG), regardless of ANN recall.\n'
    text+='\nCIs use 20,000 paired within-group query resamples, percentile 95% intervals. Saving is 100 times (1 minus ratio of mean work), not the mean of per-query percentages. Both methods use saved effective distance computations, including the CURA probe and independent selected search when ef>10. Latencies are saved per-query search means; p95 is across those query means, not request-level tail latency. Model inference, feature processing and end-to-end overhead are absent.\n'
    text+='\nPaired work, nDCG and violation tests are in statistical_tests.csv; sparse violation discordances cannot establish equivalent risk. All tests are exploratory and unadjusted. Query grouping uses reference scores only for retrospective analysis. Intervals condition on the saved policy; they exclude training/calibration uncertainty and timing-run uncertainty. The paired sign-flip test assumes sign exchangeability; the exact sign test separately assesses direction among non-ties.\n'
    (OUT/'exact_ndcg_stratified_summary.md').write_text(text,encoding='utf-8')
    return result


def frontier(budget, fixed, cura, split, calibration, split_name='MSMARCO_TEST'):
    ids = split['test_queries'] if split_name=='MSMARCO_TEST' else split['ood_queries']
    data=budget[budget.query_key.isin(ids)]
    rows=[]
    for ef,g in data.groupby('efSearch',sort=True):
        assert set(g.query_key)==set(ids)
        r=metric_row(g,f'Fixed ef={ef}',split_name)
        r['diagnostic_only']=True;r['calibration_selected_baseline']=int(ef)==96
        rows.append(r)
    cr=metric_row(cura,'CURA-Full',split_name,True)
    cr['diagnostic_only']=False;cr['calibration_selected_baseline']=False
    rows.append(cr); frame=pd.DataFrame(rows)
    eligible=frame[(frame.method!='CURA-Full')&(frame.IR_violation_rate<=cr['IR_violation_rate'])]
    cheapest=eligible.sort_values(['mean_distance_computations','efSearch']).iloc[0]
    selected=data[data.efSearch==int(cheapest.efSearch)][['query_key','distance_computations']].merge(cura[['query_key','effective_distance_computations']],on='query_key',validate='one_to_one')
    match=paired_boot(selected.effective_distance_computations,selected.distance_computations,SEED+80)
    add_test(split_name,'CURA-Full minus post-hoc risk-matched fixed','effective distance computations',selected.effective_distance_computations-selected.distance_computations,SEED+81,match['difference_ci'])
    quality=data[data.efSearch==int(cheapest.efSearch)][['query_key','nDCG@10','nDCG_loss']].merge(cura[['query_key','nDCG@10','nDCG_loss']],on='query_key',suffixes=('_fixed','_cura'),validate='one_to_one')
    quality['fixed_unsafe']=(quality.nDCG_loss_fixed>TOL).astype(int)
    quality['cura_unsafe']=(quality.nDCG_loss_cura>TOL).astype(int)
    add_test(split_name,'post-hoc risk-matched fixed minus CURA-Full','nDCG@10',quality['nDCG@10_fixed']-quality['nDCG@10_cura'],SEED+82)
    failed_fixed=set(quality.loc[quality.fixed_unsafe==1,'query_key'])
    failed_cura=set(quality.loc[quality.cura_unsafe==1,'query_key'])
    shared_failures=len(failed_fixed&failed_cura)
    suffix='' if split_name=='MSMARCO_TEST' else '_trec_saved_full'
    selected.merge(quality,on='query_key',validate='one_to_one').to_csv(OUT/f'risk_matched_fixed_per_query{suffix}.csv',index=False)
    frame.to_csv(OUT/f'fixed_risk_work_frontier{suffix}.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax in axes:
        ff=frame[frame.method!='CURA-Full']
        for _,r in ff.iterrows():
            color='#9b682b' if r.calibration_selected_baseline else '#526579'
            ax.errorbar(r.IR_violation_rate*100,r.mean_distance_computations,xerr=np.array([[r.IR_violation_rate-r.wilson95_low],[r.wilson95_upper-r.IR_violation_rate]])*100,fmt='o',color=color,alpha=.7,capsize=2)
            ax.annotate(f'{int(r.efSearch)}',(r.IR_violation_rate*100,r.mean_distance_computations),xytext=(5,3),textcoords='offset points',fontsize=8)
        ax.scatter(cr['IR_violation_rate']*100,cr['mean_distance_computations'],marker='*',s=190,color='#138d82',label='CURA-Full (probe included)',zorder=10)
        ax.axvline(cr['IR_violation_rate']*100,color='#138d82',ls=':',alpha=.5)
        ax.set_xlabel('Observed IR violation rate (%)');ax.set_ylabel('Mean distance computations');ax.grid(alpha=.15)
    axes[0].set_title('Risk and work; horizontal 95% Wilson intervals')
    axes[1].set_xlim(-.12,max(cr['IR_violation_rate']*100+1,2.2));axes[1].set_title('Observed-risk neighborhood (zoom)');axes[1].legend(fontsize=8)
    fig.suptitle(f'{split_name}: all saved fixed budgets; labels are efSearch\nPost-hoc diagnostic, not a selected deployment policy',fontsize=11)
    fig.tight_layout();savefig(fig,f'fixed_risk_work_frontier{suffix}.png')
    answer=dict(split=split_name,cura_observed_risk=cr['IR_violation_rate'],cheapest_fixed_ef=int(cheapest.efSearch),smallest_eligible_fixed_ef=int(eligible.efSearch.min()),cheapest_fixed_risk=float(cheapest.IR_violation_rate),cheapest_fixed_mean_work=float(cheapest.mean_distance_computations),cura_mean_work=cr['mean_distance_computations'],fixed_saving_vs_cura=match,shared_failure_queries=shared_failures,fixed_only_failure_queries=len(failed_fixed-failed_cura),cura_only_failure_queries=len(failed_cura-failed_fixed))
    if split_name=='MSMARCO_TEST':
        calrows=[]
        for ef,g in budget[budget.query_key.isin(split['calibration_queries'])].groupby('efSearch'):
            r=metric_row(g,f'Fixed ef={ef}','MSMARCO_CALIBRATION')
            old=calibration[calibration.efSearch==ef].iloc[0]
            assert r['IR_violations']==old.violations
            np.testing.assert_allclose(r['wilson95_upper'],old.wilson_upper,atol=1e-12)
            r['passes_Wilson_SLA']=r['wilson95_upper']<=.05
            calrows.append(r)
        cc=pd.DataFrame(calrows);cc.to_csv(OUT/'calibration_fixed_frontier.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot(cc.mean_efSearch,cc.IR_violation_rate*100,'o-',label='Observed risk')
        axes[0].plot(cc.mean_efSearch,cc.wilson95_upper*100,'s--',label='95% Wilson upper')
        axes[0].axhline(5,color='#9b682b',ls=':',label='5% SLA')
        axes[0].axvline(96,color='#138d82',ls=':');axes[0].set_xlabel('Fixed efSearch');axes[0].set_ylabel('IR violation rate (%)');axes[0].legend(fontsize=8)
        axes[1].plot(cc.IR_violation_rate*100,cc.mean_distance_computations,'o-')
        for _,r in cc.iterrows():axes[1].annotate(str(int(r.efSearch)),(r.IR_violation_rate*100,r.mean_distance_computations),fontsize=8)
        axes[1].set_xlabel('Observed calibration IR violation rate (%)');axes[1].set_ylabel('Mean distance computations')
        fig.suptitle('Calibration frontier: ef=96 is the first budget passing Wilson-SLA');fig.tight_layout();savefig(fig,'calibration_fixed_frontier.png')
        text='# Fixed-budget risk/work frontier\n\n**Post-hoc finding:** '+f"fixed ef={int(cheapest.efSearch)} is the cheapest setting by measured mean work with observed held-out risk <= CURA-Full: {cheapest.IR_violation_rate:.2%} versus {cr['IR_violation_rate']:.2%}. It uses {cheapest.mean_distance_computations:.2f} distances/query versus {cr['mean_distance_computations']:.2f}, {match['saving_pct']:.2f}% less work (paired query bootstrap 95% CI [{match['saving_pct_ci'][0]:.2f}, {match['saving_pct_ci'][1]:.2f}]%).\n\n"
        text+='This is a retrospective comparison on test outcomes, not model selection or an independently validated fixed policy. Observed-risk matching is not a population risk guarantee or an equivalence test. The original calibration-selected Fixed-IR remains ef=96.\n\n'+md(frame,['method','query_count','IR_violations','IR_violation_rate','wilson95_low','wilson95_upper','ANN_Recall_at_10','mean_nDCG_at_10','mean_distance_computations','mean_latency_ms','p95_latency_ms'])
        text+='\n\n## Why ef=96 was selected\n\nThe protocol selected the smallest fixed ef with a two-sided 95% Wilson interval upper endpoint <= 0.05 on the 200 calibration queries. At ef=64 there were 4/200 violations (2% observed), but the Wilson upper endpoint was 0.0502878655, just above the SLA. At ef=96 there was 1/200 (0.5%), upper endpoint 0.0277743999. ef=48 had 5/200, upper endpoint 0.0571791442. Therefore ef=96 was the first eligible candidate. These values were recomputed and checked against fixed_budget_calibration.csv, rather than using the test frontier to recalibrate. The confidence interval is a marginal interval, not a simultaneous or distribution-free guarantee after sweeping settings.\n\nLatency p95 is across saved per-query means; CURA sums saved probe and selected-search means and excludes controller overhead. Work includes the probe. Risk intervals use z=1.96 to reproduce the existing protocol.\n'
        (OUT/'fixed_frontier_summary.md').write_text(text,encoding='utf-8')
    return answer


def oracle_gaps(oracle, reference, budget, split):
    oo=oracle[(oracle.IR_tolerance==TOL)&(oracle.analysis_source=='ALL')].copy()
    assert oo.query_key.is_unique
    oo=oo.merge(reference[['query_key','exact_nDCG@10']],on='query_key',validate='one_to_one')
    # Verify first-crossing minima, including the legacy maximum-grid fallback.
    for key,g in budget.groupby('query_key'):
        row=oo[oo.query_key==key].iloc[0]
        for criterion,col,flag in [(g['ANN_Recall@10']>=.9,'min_ANN_safe_ef','ANN_safe_found'),(g.nDCG_loss<=TOL,'min_IR_safe_ef','IR_safe_found')]:
            candidates=g.loc[criterion,'efSearch']
            assert int(row[flag])==int(len(candidates)>0)
            assert row[col]==(candidates.min() if len(candidates) else max(EFS))
    rows=[]
    cohorts=[('MSMARCO_ALL',oo[oo.source=='msmarco-dev-small']),('MSMARCO_TEST',oo[oo.query_key.isin(split['test_queries'])]),('TREC_DL_2019',oo[oo.source=='trec-dl-2019'])]
    for cohort,df in cohorts:
        for group in ['all','exact_zero','exact_positive']:
            g=df if group=='all' else df[df['exact_nDCG@10'].eq(0) if group=='exact_zero' else df['exact_nDCG@10'].gt(0)]
            feasible=g[(g.ANN_safe_found==1)&(g.IR_safe_found==1)]
            for mode,h in [('saved_grid_censored_fallback',g),('both_criteria_feasible',feasible)]:
                row=dict(cohort=cohort,exact_group=group,estimator=mode,query_count=len(h),stratum_query_count=len(g),ANN_not_safe_in_grid=int((g.ANN_safe_found==0).sum()),IR_not_safe_in_grid=int((g.IR_safe_found==0).sum()),IR_tolerance=TOL)
                if len(h):
                    seed=SEED+100+len(rows)
                    s=paired_boot(h.min_ANN_safe_ef,h.min_IR_safe_ef,seed)
                    lower=(h.min_IR_safe_ef<h.min_ANN_safe_ef).to_numpy(float)
                    rng=np.random.default_rng(seed+1)
                    lo,hi=ci(lower[rng.integers(0,len(h),(B,len(h)))].mean(axis=1))
                    row.update(mean_ANN_safe_ef=h.min_ANN_safe_ef.mean(),mean_IR_safe_ef=h.min_IR_safe_ef.mean(),mean_ANN_minus_IR_gap=s['difference'],gap_ci95_low=s['difference_ci'][0],gap_ci95_high=s['difference_ci'][1],IR_allows_lower_rate=lower.mean(),lower_rate_ci95_low=lo,lower_rate_ci95_high=hi,relative_oracle_budget_reduction_pct=s['saving_pct'],reduction_ci95_low_pct=s['saving_pct_ci'][0],reduction_ci95_high_pct=s['saving_pct_ci'][1])
                    if group=='exact_positive':add_test(cohort+' '+mode,'ANN-safe minus IR-safe','oracle efSearch gap',h.min_ANN_safe_ef-h.min_IR_safe_ef,seed+2,s['difference_ci'])
                rows.append(row)
    result=pd.DataFrame(rows);result.to_csv(OUT/'oracle_budget_gap_analysis.csv',index=False)
    oo.to_csv(OUT/'oracle_budget_gap_per_query.csv',index=False)
    f=result[(result.cohort.isin(['MSMARCO_ALL','TREC_DL_2019']))&(result.exact_group!='all')&(result.estimator=='both_criteria_feasible')&(result.query_count>0)]
    fig,(ax,gapax)=plt.subplots(1,2,figsize=(12,5));xx=np.arange(len(f))
    ax.bar(xx-.17,f.mean_ANN_safe_ef,.34,label='ANN-safe (Recall@10 >= 0.9)',color='#526579')
    ax.bar(xx+.17,f.mean_IR_safe_ef,.34,label='IR-safe (loss <= 0.025)',color='#138d82')
    labels=[f"{'MS MARCO' if r.cohort=='MSMARCO_ALL' else 'TREC 2019'}\n{'exact-zero' if r.exact_group=='exact_zero' else 'exact-positive'}\nn={r.query_count}/{r.stratum_query_count}" for r in f.itertuples()]
    ax.set_xticks(xx,labels)
    ax.set_ylabel('Mean first-safe efSearch');ax.set_title('Both criteria feasible');ax.legend(fontsize=8)
    for j,mode in enumerate(['saved_grid_censored_fallback','both_criteria_feasible']):
        gg=result[(result.cohort.isin(['MSMARCO_ALL','TREC_DL_2019']))&(result.exact_group!='all')&(result.estimator==mode)&(result.query_count>0)]
        gapax.errorbar(xx+(j-.5)*.1,gg.mean_ANN_minus_IR_gap,yerr=np.array([gg.mean_ANN_minus_IR_gap-gg.gap_ci95_low,gg.gap_ci95_high-gg.mean_ANN_minus_IR_gap]),fmt=['o','s'][j],capsize=4,label=['Saved ef256 fallback','Both feasible'][j])
    gapax.axhline(0,color='gray',ls=':');gapax.set_xticks(xx,labels);gapax.set_ylabel('Mean ANN-safe minus IR-safe ef');gapax.set_title('Paired query bootstrap 95% intervals');gapax.legend(fontsize=8)
    fig.suptitle('Oracle budget gap: full MS MARCO cohort and 43 TREC queries\nTREC exact-zero n=0; fallback values are not verified safe minima',fontsize=11)
    fig.tight_layout()
    savefig(fig,'oracle_budget_gap_analysis.png')
    crossings=[]
    for key,g in budget.groupby('query_key'):
        g=g.sort_values('efSearch')
        crossings.append(dict(query_key=key,source=g.source.iloc[0],IR_safe_then_unsafe=bool(np.any(np.diff((g.nDCG_loss<=TOL).astype(int))<0)),ANN_safe_then_unsafe=bool(np.any(np.diff((g['ANN_Recall@10']>=.9).astype(int))<0))))
    cross=pd.DataFrame(crossings);cross.to_csv(OUT/'oracle_nonmonotonicity.csv',index=False)
    text='# Why oracle savings exist\n\n'+md(result,['cohort','exact_group','estimator','query_count','ANN_not_safe_in_grid','IR_not_safe_in_grid','mean_ANN_safe_ef','mean_IR_safe_ef','mean_ANN_minus_IR_gap','gap_ci95_low','gap_ci95_high','IR_allows_lower_rate','relative_oracle_budget_reduction_pct'])
    text+='\n\nANN safety reconstructs at least 90% of exact vector neighbors. IR safety preserves judged, rank-discounted gain within absolute nDCG loss 0.025. Missing an irrelevant or unjudged exact neighbor may reduce ANN recall without reducing the measured IR score; reordering or recovering a few judged positives can preserve utility. This mismatch can persist when exact nDCG is positive. When the exact reference is zero, nonnegative one-sided loss makes every ranking IR-safe by construction. These are mechanisms of the metric, not proof of actual relevance preservation.\n\n'
    text+='The saved first-crossing oracle substitutes ef=256 if no safe candidate exists. We report those explicitly flagged, grid-censored legacy summaries for traceability, plus summaries restricted to pairs where both criteria are feasible. A fallback of 256 is not a demonstrated safe minimum. First-safe means any tested budget meeting the criterion, not safety at every larger budget; oracle_nonmonotonicity.csv records reversals. Budget reduction is 100*(1-mean IR ef/mean ANN ef); ef reduction is not distance-computation reduction.\n\n'
    text+='The all-MS-MARCO analysis uses 1,000 queries, with a separate 200-query held-out sensitivity. TREC uses all 43 queries and has no exact-zero stratum; its empty row is NA, not a zero effect. Paired query bootstrap intervals use 20,000 resamples. Paired randomization p-values are exploratory; for TREC inspect the number of nonzero pairs and CI before interpreting them. TREC-vs-MS-MARCO differences also confound query distribution and corpus coverage, so they cannot identify a causal effect of qrel density. The same-ranking thinning experiment is needed.\n\nNonmonotonic first-safe reversals:\n\n'+md(cross.groupby('source',as_index=False)[['IR_safe_then_unsafe','ANN_safe_then_unsafe']].sum())+'\n'
    (OUT/'oracle_budget_gap_summary.md').write_text(text,encoding='utf-8')
    return result


def prepare_density():
    from prepare_trec_judgment_density import main as prepare
    return prepare()


def main():
    OUT.mkdir(exist_ok=True)
    reference=read(RESEARCH/'exact_query_reference.csv')
    budget=read(RESEARCH/'top10_query_budget_results.csv')
    oracle=read(RESEARCH/'top10_oracle_query_results.csv')
    fixed=read(CTRL/'selected_fixed_ir_msmarco_test.csv')
    cura=read(CTRL/'selected_cura_full_msmarco_test.csv')
    fixed_trec=read(CTRL/'selected_fixed_ir_trec_ood.csv')
    cura_trec=read(CTRL/'selected_cura_full_trec_ood.csv')
    calibration=read(CTRL/'fixed_budget_calibration.csv')
    split_path=CTRL/'query_split.json'
    INPUTS[split_path.relative_to(ROOT).as_posix()]=hashlib.sha256(split_path.read_bytes()).hexdigest()
    split=json.loads(split_path.read_text())
    assert reference.query_key.is_unique
    assert not budget.duplicated(['query_key','efSearch']).any()
    assert sorted(budget.efSearch.unique())==EFS
    assert len(split['test_queries'])==200 and len(split['calibration_queries'])==200
    assert not (set(split['train_queries'])&set(split['test_queries']) or set(split['test_queries'])&set(split['calibration_queries']) or set(split['train_queries'])&set(split['calibration_queries']))
    for f,ids in [(fixed,split['test_queries']),(cura,split['test_queries']),(fixed_trec,split['ood_queries']),(cura_trec,split['ood_queries'])]:validate_selection(f,budget,ids)
    check=budget.merge(reference[['query_key','exact_nDCG@10']],on='query_key',validate='many_to_one')
    np.testing.assert_allclose(check.nDCG_loss,np.maximum(0,check['exact_nDCG@10']-check['nDCG@10']),atol=1e-12)
    for f in [fixed,fixed_trec]:
        np.testing.assert_allclose(f.effective_distance_computations,f.distance_computations)
        np.testing.assert_allclose(f.effective_latency_ms,f.mean_latency_ms)
    for f in [cura,cura_trec]:
        np.testing.assert_allclose(f.effective_distance_computations,np.where(f.efSearch==10,f.probe_ndis,f.probe_ndis+f.distance_computations))
        np.testing.assert_allclose(f.effective_latency_ms,np.where(f.efSearch==10,f.probe_latency_ms,f.probe_latency_ms+f.mean_latency_ms))
        probecheck=f[['query_key','probe_ndis']].merge(budget[budget.efSearch==10][['query_key','distance_computations']],on='query_key',validate='one_to_one')
        np.testing.assert_allclose(probecheck.probe_ndis,probecheck.distance_computations)
    ss=exact_split(reference,fixed,cura)
    ff=frontier(budget,fixed,cura,split,calibration)
    oo=oracle_gaps(oracle,reference,budget,split)
    print('Parts 1-3 completed from saved tables; no retrieval executed.',flush=True)
    tf=frontier(budget,fixed_trec,cura_trec,split,calibration,'TREC_DL_2019')
    audit=prepare_density()
    INPUTS.update(audit['source_hashes'])
    pd.DataFrame(STATS).to_csv(OUT/'statistical_tests.csv',index=False)
    from extend_statistics import main as extend_statistics
    extend_statistics()
    for filename in ['05_research_grade_validation.py','07_cura_controller.py','09_final_ablation_package.py']:
        INPUTS[filename]=hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
    # Verify immutable inputs after the complete run.
    for path,digest in INPUTS.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    protocol=dict(seed=SEED,bootstrap_resamples=B,permutation_resamples=B,CI='paired-query percentile bootstrap 95%; conditional on saved policies and post-hoc comparator',IR_tolerance=TOL,ANN_target=.9,query_identity_checks='passed',saved_selection_metric_checks='passed',input_hashes=INPUTS,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,original_inputs_unchanged=True,judgment_density_status=audit['status'],qrels_recovered=True,qrel_seeds_prepared=50,ranking_seeds_evaluated=0,retrieval_executed=False)
    (OUT/'analysis_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    def records(f):return f.astype(object).where(pd.notna(f),None).to_dict('records')
    answers=dict(exact_split=records(ss),fixed_frontier=ff,trec_saved_full_frontier=tf,oracle=records(oo),judgment_density_status=audit['status'],qrels_recovered=True)
    (OUT/'analysis_results.json').write_text(json.dumps(answers,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(split=ss[['exact_group','method','query_count','CURA_work_saving_vs_fixed_pct','contribution_to_total_work_reduction_pct']].to_dict('records'),fixed=ff,trec=tf,thinning='BLOCKED'),indent=2))


if __name__=='__main__':main()

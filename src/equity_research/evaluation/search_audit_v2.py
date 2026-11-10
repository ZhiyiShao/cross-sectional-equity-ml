"""Describe saved V2 development candidates; no model or market-data imports."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FOLDERS = ('context_selection_v2', 'ranking_selection_v2')
DESIGNS = ('ridge_original', 'ridge_enhanced', 'lightgbm_original',
           'lightgbm_enhanced', 'ranker_original', 'ranker_enhanced')
PAIRS = (
    ('ridge_features', 'ridge_original', 'ridge_enhanced'),
    ('regression_features', 'lightgbm_original', 'lightgbm_enhanced'),
    ('ranking_features', 'ranker_original', 'ranker_enhanced'),
    ('original_objective', 'lightgbm_original', 'ranker_original'),
    ('enhanced_objective', 'lightgbm_enhanced', 'ranker_enhanced'),
)


def summarize(frame):
    """Require the frozen 6 x 4 x 2 design, then retain equal window weights."""
    if len(frame) != 48 or set(frame.design) != set(DESIGNS):
        raise ValueError('Expected all 48 development candidate-window records')
    if frame.duplicated(['design', 'candidate', 'window']).any():
        raise ValueError('Duplicate candidate-window record')
    if not np.isfinite(frame[['rank_ic', 'parameter']]).all().all():
        raise ValueError('Nonfinite candidate values')
    for design, group in frame.groupby('design'):
        grid = [0.1, 10., 1000., 100000.] if design.startswith('ridge') else [7., 15., 31., 63.]
        if set(group.window) != {1, 2}:
            raise ValueError('Missing development window')
        for window, rows in group.groupby('window'):
            rows = rows.sort_values('candidate')
            if rows.candidate.tolist() != [0, 1, 2, 3] or rows.parameter.tolist() != grid:
                raise ValueError('Incomplete or changed parameter grid')
            if not rows.dates.eq({1: 49, 2: 100}[window]).all():
                raise ValueError('Changed development date coverage')
            if not rows.rows.eq({1: 24427, 2: 49860}[window]).all():
                raise ValueError('Changed development sample coverage')
    curve = frame.pivot(index=['design', 'candidate', 'parameter'], columns='window',
                        values='rank_ic').rename(columns={1: 'ic_2019', 2: 'ic_2020_2021'}).reset_index()
    curve.columns.name = None
    curve['equal_window_ic'] = (curve.ic_2019 + curve.ic_2020_2021) / 2
    summaries = []
    for design, group in curve.groupby('design', sort=False):
        ordered = group.sort_values(['equal_window_ic', 'candidate'], ascending=[False, True])
        best, runner = ordered.iloc[0], ordered.iloc[1]
        win1 = group.sort_values(['ic_2019', 'candidate'], ascending=[False, True]).iloc[0]
        win2 = group.sort_values(['ic_2020_2021', 'candidate'], ascending=[False, True]).iloc[0]
        summaries.append(dict(
            design=design, selected_parameter=best.parameter,
            boundary='lower' if best.candidate == 0 else 'upper' if best.candidate == 3 else 'interior',
            equal_window_ic=best.equal_window_ic, runner_up_parameter=runner.parameter,
            winner_margin=best.equal_window_ic-runner.equal_window_ic,
            best_2019=win1.parameter, best_2020_2021=win2.parameter,
            window_candidate_rank_correlation=group.ic_2019.corr(group.ic_2020_2021, method='spearman'),
            selected_regret_2019=win1.ic_2019-best.ic_2019,
            selected_regret_2020_2021=win2.ic_2020_2021-best.ic_2020_2021))
    contrasts = []
    for contrast, baseline, treatment in PAIRS:
        left = curve[curve.design.eq(baseline)].set_index('parameter')
        right = curve[curve.design.eq(treatment)].set_index('parameter')
        for parameter in left.index:
            delta1 = right.loc[parameter, 'ic_2019']-left.loc[parameter, 'ic_2019']
            delta2 = right.loc[parameter, 'ic_2020_2021']-left.loc[parameter, 'ic_2020_2021']
            contrasts.append(dict(contrast=contrast, baseline=baseline, treatment=treatment,
                                  parameter=parameter, delta_2019=delta1, delta_2020_2021=delta2,
                                  equal_window_delta=(delta1+delta2)/2,
                                  positive_both_windows=bool(delta1 > 0 and delta2 > 0)))
    return curve, pd.DataFrame(summaries), pd.DataFrame(contrasts)


def main():
    # Explicit allowlist: only aggregate development CSVs can be read here.
    sources = [ROOT/'results'/folder/name for folder in FOLDERS
               for name in ('candidate_metrics.csv', 'candidate_ranking.csv')]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    frame = pd.concat([pd.read_csv(sources[0]), pd.read_csv(sources[2])], ignore_index=True)
    curve, summary, contrasts = summarize(frame)
    saved = pd.concat([pd.read_csv(sources[1]), pd.read_csv(sources[3])])
    check = curve.merge(saved, on=['design', 'candidate', 'parameter'], validate='one_to_one')
    if len(check) != 24:
        raise ValueError('Saved candidate ranking coverage mismatch')
    np.testing.assert_allclose(check.equal_window_ic, check.rank_ic, atol=1e-14, rtol=0)
    out = ROOT/'results/search_audit_v2'
    out.mkdir(exist_ok=True)
    for name, table in [('candidate_curves', curve), ('search_summary', summary), ('matched_contrasts', contrasts)]:
        table.to_csv(out/f'{name}.csv', index=False, lineterminator='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharey=True)
    for ax, design in zip(axes.flat, DESIGNS):
        group = curve[curve.design.eq(design)].sort_values('candidate')
        for col, label in [('ic_2019', '2019'), ('ic_2020_2021', '2020-2021'),
                           ('equal_window_ic', 'Equal-window mean')]:
            ax.plot(range(4), group[col], marker='o', label=label)
        ax.axhline(0, color='gray', lw=.7)
        ax.set_xticks(range(4), [f'{p:g}' for p in group.parameter])
        ax.set_xlabel('alpha (categorical grid)' if design.startswith('ridge') else 'num_leaves (categorical grid)')
        ax.set_ylabel('Mean Rank IC'); ax.set_title(design); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('V2 saved development candidates | no refits | 2019 and 2020-2021 only')
    fig.tight_layout(); fig.savefig(out/'candidate_curves.png', dpi=160); plt.close(fig)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    audit = dict(source_sha256=hashes, candidate_window_rows=48, candidate_configurations=24,
                 matched_contrasts=20, model_fits=0, market_data_reads=0,
                 holdout_performance_reads=0, saved_ranking_reproduced=True,
                 source_files_unchanged=True, selection_rule_changed=False,
                 inference='Descriptive aggregate audit; no significance test or new model selection.')
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

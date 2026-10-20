"""Reproducible, score-independent test event audit. Unresolved terms fail closed."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from equity_research.models.common import sha
import equity_research.backtest.market as base

from equity_research.paths import ROOT
OUT = ROOT / 'outputs/test_event_audit_v1'
# Final legal ratios; no inference from vendor dollar values or model outcomes.
STOCK = {
    76201: (1.7234, 'https://ir.amd.com/financial-information/sec-filings/content/0000002488-22-000031/xilinxacquisitionclosepres.htm'),
    90805: (1., 'https://www.sec.gov/Archives/edgar/data/1437107/000119312522103051/d328161d8k.htm'),
    13169: (.0406, 'https://www.sec.gov/Archives/edgar/data/946581/000119312522160004/d306239dex999.htm'),
    62341: (.4638, 'https://www.sec.gov/Archives/edgar/data/93410/000119312523170147/d512919ds4.htm'),
    19752: (.84, 'https://d1io3yog0oux5.cloudfront.net/_e36a86a1f2c16825e6800bc362b5124f/exxonmobil/db/2427/22271/pdf/Denbury_Form_8937_-_2023.pdf'),
    15785: (.03002, 'https://investors.jmsmucker.com/files/doc_financials/2024/ar/FY24_AnnualReport__FINAL-web.pdf'),
    75241: (2.3234, 'https://investor.exxonmobil.com/sec-filings/all-sec-filings/content/0000950103-24-006322/dp210867_8k.htm'),
    21186: (1., 'https://www.sec.gov/Archives/edgar/data/2005951/000110465924087515/tm2421045d1_ex99-1.htm'),
    15069: (.255, 'https://www.sec.gov/Archives/edgar/data/101778/000110465924122113/tm2428994d1_8k.htm'),
    92121: (1.0192, 'https://www.sec.gov/Archives/edgar/data/927628/000119312525122059/d934475d8k.htm'),
    19409: (1.034, 'https://www.sec.gov/Archives/edgar/data/1782754/000119312525154115/d58643dex991.htm'),
    83621: (.3399, 'https://www.sec.gov/Archives/edgar/data/1013462/000114036125026140/ef20051970_425.htm'),
    28484: (1.025, 'https://www.sec.gov/Archives/edgar/data/1789832/000119312525161221/d66017d8k.htm'),
    81593: (11., 'https://www.rocketcompanies.com/press-release/mr-cooper-americas-largest-servicer-joins-rocket-the-nations-largest-lender/'),
}
CVR = {
    75107: (35., 'https://www.jnj.com/media-center/press-releases/johnson-johnson-completes-acquisition-of-abiomed'),
    14054: (12., 'https://www.bms.com/investors/shareholder-services/mirati-faqs.html'),
    15284: (6., 'https://www.sanofi.com/en/media-room/press-releases/2025/2025-07-18-05-00-00-3117709'),
}

def possible_held_events(raw, candidates, schedule):
    """Union of all possible selected holdings, including unsellable carryovers."""
    candidates = candidates[candidates.signal_eligible]
    indexed = {r.entry_date: set(candidates.loc[candidates.signal_date.eq(r.signal_date), 'PERMNO'])
               for r in schedule.itertuples()}
    held = set()
    result = []
    tickers = raw.dropna(subset=['Ticker']).query("Ticker != ''").groupby('PERMNO').Ticker.last()
    for date, frame in raw.groupby('DlyCalDt', sort=True):
        terminal = set(frame.loc[frame.DlyDelFlg.eq('Y'), 'PERMNO'])
        for p in sorted(held & terminal):
            result.append(dict(PERMNO=int(p), date=str(date.date()), ticker=tickers.get(p, '')))
        held -= terminal
        if date in indexed:
            sellable = set(frame.loc[frame.sellable, 'PERMNO'])
            buyable = set(frame.loc[frame.buyable, 'PERMNO'])
            held = (held - sellable) | (indexed[date] & buyable)
    return pd.DataFrame(result)

def main():
    d = pd.read_csv(OUT / 'wrds_delistings.csv', keep_default_na=False)
    s = pd.read_csv(OUT / 'wrds_distributions.csv', keep_default_na=False)
    raw = pd.read_parquet(OUT / 'candidate_market_local.parquet')
    candidates = pd.read_parquet(OUT / 'candidates_local.parquet')
    schedule = pd.read_csv(OUT / 'schedule.csv', parse_dates=['signal_date', 'entry_date', 'end_date'])
    held = possible_held_events(raw, candidates, schedule)
    held.to_csv(OUT / 'potential_held_events_local.csv', index=False)
    assert len(d) == 96 and not d.permno.duplicated().any()
    assert set(d.permno) == set(raw.loc[raw.DlyDelFlg.eq('Y'), 'PERMNO'])
    assert set(held.PERMNO) <= set(d.permno)
    events = []
    for r in d[d.permno.isin(held.PERMNO)].sort_values(['deldlydt', 'permno']).itertuples():
        z = s[s.permno.eq(r.permno) & s.disexdt.eq(r.deldlydt)]
        assert set(z.distype) <= {'CP', 'SP', 'SD'}
        cash = float(pd.to_numeric(z.loc[z.distype.eq('CP'), 'disdivamt']).sum())
        special = float(pd.to_numeric(z.loc[z.distype.eq('SD'), 'disdivamt']).sum())
        ratio, source = STOCK.get(r.permno, (0., 'WRDS queries 11808005 and 11808011'))
        status = 'verified_terms'; policy = 'fixed_terms'; successor = int(r.delpermno)
        if r.permno == 92257:
            cash = 142.5; ratio = 0.; successor = 0; policy = 'no_election_cash'
            source = 'https://www.broadcom.com/company/news/financial-releases/61501'
        cvr_max = 0.
        if r.permno in CVR:
            cvr_max, source = CVR[r.permno]; policy = 'cvr_excluded_from_investable_nav'
        if r.permno in {11786, 90090, 12448}:
            policy = 'first_post_delisting_quote_forced_sale_proxy'
        if successor:
            assert ratio > 0, f'Missing legal stock ratio {r.permno}'
        if policy == 'fixed_terms':
            assert abs(pd.to_numeric(z.disdivamt).sum() - float(r.deldivamt)) < 1e-5
        pay = z.loc[z.distype.eq('SD'), 'dispaydt'].tolist()
        events.append(dict(permno=int(r.permno), ticker=held.set_index('PERMNO').loc[r.permno, 'ticker'],
            event_date=r.deldlydt, cash_per_share=cash, special_per_share=special,
            special_expected_date=max(pay) if pay else None, ratio=ratio, successor_permno=successor,
            source=source, election_policy=policy, status=status, cvr_max_per_share=cvr_max,
            previous_price=float(r.deldtprc), vendor_value=float(r.deldivamt),
            vendor_return=r.delret, vendor_return_flag=r.delretmisstype,
            vendor_payment_dates=sorted(set(z.dispaydt)),
            otc_first_observation_date=r.delnextdt or None,
            otc_first_observation_price=float(r.delnextprc) if r.delnextprc else None))
    # Explicit user approval, received before any test model prediction or performance.
    result = dict(status='terms_verified_execution_audit_pending', date_scope=['2022-01-01','2025-12-24'],
        exception_authorization='采用简化口径并明确披露',
        exception_rules=dict(otc='Freeze last held value; on CRSP delnextdt mark to delnextprc and force sell with normal cost and settlement. Quote is an execution proxy, not confirmed fill. No earlier use of future quote.',
            cvr='Record entitlement quantity and contractual maximum; excluded from investable NAV and buying power. Not a fair-value estimate; no claim that the right is worthless.'),
        query_ids=[11808005,11808011], events=events, inputs={
            str(p.relative_to(ROOT)): sha(p) for p in [OUT/'wrds_delistings.csv', OUT/'wrds_distributions.csv']},
        unresolved_count=sum(e['status'] != 'verified_terms' for e in events),
        no_predictions_or_performance_used=True)
    (OUT/'action_registry_draft.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame(events).to_csv(OUT/'action_terms_local.csv', index=False)
    print(json.dumps(dict(possible_held_events=len(events), unresolved=result['unresolved_count'],
                         distributions=len(s), model_fits=0, predictions=0)))
    extra = sorted({e['successor_permno'] for e in events if e['ratio']} - set(raw.PERMNO))
    columns = [c for c in raw if c not in ['sellable','buyable','ordinary_return','known_terminal']]
    source = ROOT/'data/interim/crsp_audit_v1'
    frames = [raw]; access = []
    for name in json.loads((source/'ingest.json').read_text())['parts']:
        path = source/name; md = pq.read_metadata(path)
        lo = min(md.row_group(i).column(0).statistics.min for i in range(md.num_row_groups))
        hi = max(md.row_group(i).column(0).statistics.max for i in range(md.num_row_groups))
        wanted = [p for p in extra if lo <= p <= hi]
        if not wanted: continue
        f = pd.read_parquet(path, columns=columns, filters=[('PERMNO','in',wanted),
            ('DlyCalDt','>=',raw.DlyCalDt.min()),('DlyCalDt','<=',raw.DlyCalDt.max())])
        frames.append(f); access.append(dict(part=name,rows=len(f),permnos=wanted))
    raw = pd.concat(frames,ignore_index=True)
    uc = json.loads((ROOT/'config/universe.json').read_text())
    raw['sellable'] = base.regular_trade(raw,uc)
    raw['buyable'] = raw.sellable & base.filter_codes(raw,uc['security_filters'])
    raw['ordinary_return'] = raw.DlyDelFlg.eq('N') & raw.DlyRetMissFlg.eq('NA') & raw.DlyRetDurFlg.isin(base.SINGLE_DAY) & np.isfinite(raw.DlyRet) & raw.DlyRet.ge(-1)
    raw['known_terminal'] = raw.DlyDelFlg.eq('Y') & raw.DlyRetMissFlg.eq('NA') & raw.DlyRetDurFlg.isin(base.SINGLE_DAY+['DD']) & np.isfinite(raw.DlyRet) & raw.DlyRet.ge(-1)
    assert not raw.duplicated(['DlyCalDt','PERMNO']).any()
    for e in events:
        stock = 0.
        if e['ratio']:
            price = raw.loc[raw.PERMNO.eq(e['successor_permno']) & raw.DlyCalDt.eq(pd.Timestamp(e['event_date'])), 'DlyPrc']
            assert len(price)==1 and float(price.iloc[0])>0
            stock = e['ratio']*float(price.iloc[0])
        e['investable_conversion_value'] = stock+e['cash_per_share']+e['special_per_share']
        e['vendor_value_gap'] = e['investable_conversion_value']-e['vendor_value']
    raw.to_parquet(OUT/'execution_market_local.parquet',index=False)
    (OUT/'successor_access_log.json').write_text(json.dumps(access,indent=2),encoding='utf-8')
    (OUT/'action_registry_draft.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame(events).to_csv(OUT/'action_terms_local.csv',index=False)
    print('Successor observations added:', len(raw), 'total market rows')

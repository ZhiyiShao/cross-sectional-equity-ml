"""Build all validation delisting terms from WRDS and published stock ratios."""
import json
from pathlib import Path
import pandas as pd
from equity_research.paths import ROOT
RAW=ROOT/'data/raw/corporate_actions'
# Stock ratios are legal exchange ratios, never inferred from vendor stock value.
EXTRA={
11896:(.63,'https://investor.analog.com/news-releases/news-release-details/analog-devices-completes-acquisition-maxim-integrated/'),
12374:(2.323,'https://investor.marvell.com/sec-filings/all-sec-filings/content/0001193125-21-123305/d282209dex991.htm'),
12650:(2.884,'https://s21.q4cdn.com/736796105/files/doc_financials/2021/ar/2021-special-cp.pdf'),
14040:(.10256,'https://www.t-mobile.com/news/un-carrier/t-mobile-sprint-business-combination-agreement'),
15079:(.4125,'https://investor.iconplc.com/news-releases/news-release-details/icon-acquisition-creates-world-leading-healthcare-intelligence/'),
18734:(.0776,'https://www.sec.gov/Archives/edgar/data/1764925/000119312521220234/d171429d8k.htm'),
24942:(2.3348,'https://www.sec.gov/Archives/edgar/data/101829/000114036120007906/nc10010681x1_ex99-1.htm'),
31500:(.5833,'https://www.sec.gov/Archives/edgar/data/350797/000095010321003212/dp147036_8k.htm'),
83111:(2.1243,'https://www.astrazeneca.com/media-centre/press-releases/2020/astrazeneca-to-acquire-alexion.html'),
84597:(1.0837,'https://content.schwab.com/web/retail/public/about-schwab/SEC_Form8kA_12042020.pdf'),
90272:(3.38,'https://investors.centene.com/2020-01-23-Centene-Completes-Acquisition-of-WellCare%2C-Creating-a-Leading-Healthcare-Enterprise-Focused-on-Government-Sponsored-Healthcare-Programs?aspdf=1'),
92239:(1.46,'https://www.conocophillips.com/news-media/story/conocophillips-completes-acquisition-of-concho-resources/')}

def main():
 d=pd.read_csv(RAW/'validation_delistings.csv',keep_default_na=False)
 s=pd.read_csv(RAW/'validation_distributions.csv',keep_default_na=False)
 f=pd.read_parquet(RAW/'event_price_lookup_local.parquet')
 old={x['permno']:x for x in json.loads((ROOT/'config/corporate_actions.json').read_text())['events']}
 wanted=set(pd.read_csv(ROOT/'outputs/pretest_closeout_v1/delisting_audit_local.csv').PERMNO)
 assert set(d.permno)==wanted and len(d)==39
 events=[]
 for r in d.itertuples():
  assert '2020-01-01'<=r.deldlydt<='2021-12-31'
  z=s[s.permno.eq(r.permno)&s.disexdt.eq(r.deldlydt)]
  assert len(z) and set(z.distype)<= {'CP','SP','SD'}
  assert abs(z.disdivamt.astype(float).sum()-float(r.deldivamt))<1e-5
  cash=float(z.loc[z.distype.eq('CP'),'disdivamt'].astype(float).sum())
  special=float(z.loc[z.distype.eq('SD'),'disdivamt'].astype(float).sum())
  ratio=old.get(r.permno,{}).get('ratio',0)
  source=old.get(r.permno,{}).get('source','WRDS query 11807755 + 11807760')
  successor=int(r.delpermno);policy='fixed_terms'
  if r.permno in EXTRA:ratio,source=EXTRA[r.permno]
  if r.permno==13267:
   cash=12.41;ratio=0;successor=0;policy='no_election_cash'
   source='https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2020-128'
  if r.permno==31500:cash=28.25;policy='no_election_mixed'
  if successor:assert ratio>0
  ticker=f.loc[f.PERMNO.eq(r.permno)&f.Ticker.fillna('').ne(''),'Ticker'].iloc[-1]
  stock=0
  if ratio:
   prices=f.loc[f.PERMNO.eq(successor)&f.DlyCalDt.eq(r.deldlydt),'DlyPrc']
   assert len(prices)==1;stock=ratio*float(prices.iloc[0])
  pay=z.loc[z.distype.eq('SD'),'dispaydt'].tolist()
  events.append(dict(permno=int(r.permno),ticker=ticker,event_date=r.deldlydt,
    cash_per_share=cash,special_per_share=special,special_expected_date=max(pay) if pay else None,
    ratio=ratio,successor_permno=successor,source=source,election_policy=policy,
    vendor_value=float(r.deldivamt),economic_value=cash+special+stock,
    bridge_gap=cash+special+stock-float(r.deldivamt),previous_price=float(r.deldtprc),
    vendor_payment_dates=sorted(set(z.dispaydt)),vendor_return=float(r.delret)))
 assert len(events)==39
 (ROOT/'config/validation_corporate_actions.json').write_text(json.dumps(dict(
  date_scope=['2020-01-01','2021-12-31'],query_ids=[11807755,11807760],
  primary_availability_lag=2,availability_lags=[0,2,5,20],trade_settlement_lag=2,
  costs_bps=[0,5,10,20],primary_cost_bps=10,events=events),indent=2,ensure_ascii=False),encoding='utf-8')
 print(pd.DataFrame(events)[['ticker','ratio','cash_per_share','bridge_gap','election_policy']].to_string(index=False))

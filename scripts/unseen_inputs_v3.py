"""Rebuild scheduled cross sections and, separately, strict labels from normalized CRSP."""
import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from equity_research.data import features, universe
from equity_research.data.date_gate import signal_schedule
from equity_research.evaluation.exposures import sic_group
from equity_research.models.preprocessing import require

KEY = ["signal_date", "PERMNO"]
SEC = ["ConditionalType", "TradingStatusFlg", "DlyDelFlg", "DlyPrcFlg", "DlyPrc",
       "DlyClose", "DlyVol", "SecurityType", "SecuritySubType", "ShareType",
       "USIncFlg", "IssuerType", "PrimaryExch"]
FEATURE = SEC + ["DlyHigh", "DlyLow", "DlyCap", "ShrOut", "DlyCumFacShr", "DlyRet",
                 "DlyRetMissFlg", "DlyRetDurFlg", "SICCD", "SecInfoStartDt", "SecInfoEndDt"]


class Sources:
    """Disjoint vintage date ranges, with normalized unique security/session keys."""
    def __init__(self, root, start, end, extension=None, *, history_end=None):
        old = root/"data/interim/crsp_audit_v1"
        metadata = json.loads((old/"ingest.json").read_text())
        cutoff = pd.Timestamp(metadata["last_date"])
        if history_end is not None:
            require(pd.Timestamp(history_end)<=cutoff,"History cutoff beyond normalized archive")
            cutoff=pd.Timestamp(history_end)
        self.end = pd.Timestamp(end)
        self.start = pd.Timestamp(start)
        self.datasets = []
        calendars = []
        sources = [([old/p for p in metadata["parts"]], old/"calendar.parquet",
                    pd.Timestamp(metadata["first_date"]), min(self.end,cutoff))]
        if self.end > cutoff:
            require(extension is not None,"Missing approved normalized vintage")
            sources.append((*extension,cutoff+pd.Timedelta(days=1),self.end))
        for parts,calendar,lo,hi in sources:
            data = ds.dataset(list(map(str,parts)),format="parquet")
            self.datasets.append((data,lo,hi))
            # Original normalized calendar stores ISO dates as strings.
            cf = pd.read_parquet(calendar,columns=["date","vwretd"],
                                 filters=[("date",">=",str(lo.date())),("date","<=",str(hi.date()))])
            calendars.append(cf)
        cal = pd.concat(calendars)
        self.market = pd.Series(cal.vwretd.to_numpy(),index=pd.DatetimeIndex(pd.to_datetime(cal.date)))
        require(self.market.index.is_unique and self.market.index.is_monotonic_increasing and
                np.isfinite(self.market).all() and self.market.index.max()==self.end,"Invalid session calendar")
        self.schedule = signal_schedule(self.market.index,self.start,self.end)

    def read(self,columns,end,ids=None):
        for data,lo,hi in self.datasets:
            bound=min(pd.Timestamp(end),hi)
            if bound<lo: continue
            filt=(ds.field("DlyCalDt")>=lo)&(ds.field("DlyCalDt")<=bound)
            if ids is not None: filt &= ds.field("PERMNO").isin(ids)
            for fragment in data.get_fragments(filter=filt):
                batches=fragment.to_batches(columns=["PERMNO","DlyCalDt",*columns],filter=filt,batch_size=65536)
                for batch in batches:
                    if len(batch): yield batch.to_pandas()


def membership(source,cfg):
    cal=source.market.index
    last=pd.Series(cal,index=cal.to_period("M")).groupby(level=0).max()
    months=source.schedule.signal_date.dt.to_period("M").unique()
    formations=pd.DatetimeIndex(last.loc[months-1])
    # Aggregate counts and dollar sums across batches. Each window is in market sessions.
    observations=[]; snapshots=[]; liquidity=[]
    for g in source.read(SEC+["PERMCO"],formations.max()):
        observed=(g.DlyDelFlg.eq("N") & g.ConditionalType.eq("RW") & g.TradingStatusFlg.eq("A") &
                  g.DlyPrc.gt(0) & g.DlyPrcFlg.isin(["TR","BA"]))
        valid=g.loc[observed,["PERMNO","DlyCalDt"]]
        observations.append(valid.groupby(["PERMNO",valid.DlyCalDt.dt.to_period("M")]).size())
        recent=g.loc[g.DlyCalDt.ge(cal[cal.get_loc(formations[0])-cfg["liquidity_window_trading_days"]+1])]
        if recent.empty: continue
        trade=universe.regular_trade(recent,cfg)
        dollars=(recent.DlyClose*recent.DlyVol).where(trade)
        for day in formations:
            first=cal[cal.get_loc(day)-cfg["liquidity_window_trading_days"]+1]
            window=recent.DlyCalDt.between(first,day)
            if window.any():
                stats=dollars[window].groupby(recent.loc[window,"PERMNO"]).agg(["sum","count"])
                liquidity.append(stats.assign(formation_date=day).reset_index())
        snap=recent.loc[recent.DlyCalDt.isin(formations)].copy()
        if len(snap):
            snap["eligible_security"]=universe.filter_codes(snap,cfg["security_filters"])
            snap["eligible_trading"]=trade.loc[snap.index]
            snapshots.append(snap[["DlyCalDt","PERMNO","PERMCO","DlyClose","eligible_security","eligible_trading"]])
    hist=pd.concat(observations).groupby(level=[0,1]).sum().unstack(fill_value=0).sort_index(axis=1).cumsum(axis=1)
    adv=pd.concat(liquidity).groupby(["formation_date","PERMNO"])[["sum","count"]].sum()
    s=pd.concat(snapshots).rename(columns={"DlyCalDt":"formation_date"}).join(adv,on=["formation_date","PERMNO"])
    require(not s.duplicated(["formation_date","PERMNO"]).any(),"Duplicate normalized membership key")
    picks=[]
    for day,g in s.groupby("formation_date"):
        g=g.copy()
        g["adv60_usd"]=g["sum"]/g["count"]
        g["eligible_history"]=g.PERMNO.map(hist[day.to_period("M")]).ge(cfg["min_observed_history_days"])
        g["eligible_price"]=g.DlyClose.ge(cfg["min_close_usd"])
        g["eligible_liquidity"]=g["count"].ge(cfg["min_valid_liquidity_days"]) & g.adv60_usd.gt(0)
        selected,_=universe.select_month(g,cfg)
        require(len(selected)==cfg["max_securities"],"Incomplete monthly universe")
        picks.append(selected[["formation_date","PERMNO","PERMCO","liquidity_rank"]].assign(effective_month=str(day.to_period("M")+1)))
    result=pd.concat(picks,ignore_index=True)
    require(result.effective_month.nunique()==len(months),"Missing formation")
    return result


def panel(source,keys,cfg,*,labels=False):
    ids=sorted(keys.PERMNO.unique())
    market=source.market if labels else source.market.loc[:keys.signal_date.max()]
    columns=SEC+["DlyRet","DlyRetDurFlg","DlyRetMissFlg"] if labels else FEATURE
    out=[]
    for start in range(0,len(ids),100):
        raw=pd.concat(source.read(columns,market.index.max(),ids[start:start+100]))
        for permno,g in raw.groupby("PERMNO"):
            x=g.set_index("DlyCalDt").reindex(market.index)
            signals=keys.loc[keys.PERMNO.eq(permno),"signal_date"].sort_values()
            pos=market.index.get_indexer(signals)
            require((pos>=0).all(),"Missing signal session")
            trade=universe.regular_trade(x,cfg)
            eligible=trade & universe.filter_codes(x,cfg["security_filters"])
            if labels:
                segments=features.return_segments(x)
                entries=eligible.to_numpy(); exits=trade.to_numpy()
                f=pd.DataFrame([features.label_one(i,x,market,entries,exits,segments) for i in pos],
                               index=signals)
                f=f[["label_end_date","label_status","y_excess_5d"]]
            else:
                f=features.make_features(x,market,cfg).iloc[pos].copy()
                f.index=signals
                at=x.iloc[pos]
                f["industry"]=[sic_group(c,s,e,d) for c,s,e,d in zip(at.SICCD,at.SecInfoStartDt,at.SecInfoEndDt,signals)]
                f["signal_eligible"]=eligible.iloc[pos].to_numpy()
            out.append(f.rename_axis("signal_date").reset_index().assign(PERMNO=permno))
    result=pd.concat(out,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(result[KEY],keys[KEY].sort_values(KEY).reset_index(drop=True),check_dtype=False)
    return result


def build_features(source,cfg):
    universe_rows=membership(source,cfg)
    dates=source.schedule[["signal_date"]].assign(effective_month=source.schedule.signal_date.dt.strftime("%Y-%m"))
    keys=dates.merge(universe_rows,on="effective_month")
    require(keys.formation_date.lt(keys.signal_date).all(),"Invalid formation date")
    return universe_rows,panel(source,keys,cfg)

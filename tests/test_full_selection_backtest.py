import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from reference.portfolio import execute_rebalance, simulate, verify_primary, make_targets, MODELS, read_predictions


def market(n=6):
    shape=(n,1)
    return dict(ids=np.array([1]),dates=pd.bdate_range('2020-01-01',periods=n),
        ret=np.zeros(shape),ordinary=np.ones(shape,bool),terminal=np.zeros(shape,bool),
        buyable=np.ones(shape,bool),sellable=np.ones(shape,bool),flag=np.full(shape,'NA',object),
        span_start=np.full(shape,-1,int),valid_price=np.ones(shape,bool))


class BacktestTests(unittest.TestCase):
    def test_failed_buy_budget_stays_cash(self):
        v,c,t=execute_rebalance(np.zeros(2),1.,np.zeros(2),[0,1],np.array([True,False]),np.ones(2,bool),.001)
        self.assertAlmostEqual(v[0],.5/1.001)
        self.assertAlmostEqual(c,.5)
        self.assertAlmostEqual(t['failed_buy'][1],.5/1.001)

    def test_failed_sell_blocks_new_buy(self):
        v,c,t=execute_rebalance(np.array([1.,0.]),0.,np.zeros(2),[1],np.ones(2,bool),np.array([False,True]),.001)
        np.testing.assert_allclose(v,[1,0]);self.assertEqual(c,0)
        self.assertEqual(t['failed_sell'][0],1)

    def test_timing_fees_and_independent_ledger(self):
        m=market(4);m['ret'][:,0]=[0,9,.1,.2]
        d,o,h,e=simulate(m,{m['dates'][1]:(1,)},10,None,True)
        self.assertAlmostEqual(d.nav.iloc[-1],1/1.001*1.1*1.2*.999)
        self.assertEqual(verify_primary(d,o,h),4)
        self.assertAlmostEqual(d.nav.iloc[1],1/1.001)

    def test_delisting_freezes_and_releases_only_on_assumed_day(self):
        m=market();m['ordinary'][2:,0]=False;m['terminal'][2,0]=True
        m['ret'][2,0]=-.2;m['buyable'][2:,0]=False;m['sellable'][2:,0]=False
        d,*_=simulate(m,{m['dates'][1]:(1,)},0,None)
        self.assertAlmostEqual(d.receivable.iloc[-1],.8);self.assertEqual(d.cash.iloc[-1],0)
        d,*_=simulate(m,{m['dates'][1]:(1,)},0,2)
        self.assertEqual(d.cash.iloc[3],0);self.assertAlmostEqual(d.cash.iloc[4],.8)
        self.assertAlmostEqual(d.nav.iloc[-1],.8)

    def test_missing_price_bridge_applied_once(self):
        m=market();m['ordinary'][2:4,0]=False;m['flag'][2,0]='MP'
        m['ret'][2,0]=np.nan;m['ret'][3,0]=.2;m['span_start'][3,0]=1
        d,*_=simulate(m,{m['dates'][1]:(1,)},0,None)
        self.assertTrue(d.valuation_estimated.iloc[2]);self.assertAlmostEqual(d.nav.iloc[-1],1.2)
        self.assertTrue(np.isnan(m['ret'][2,0]))

    def test_unresolved_return_aborts(self):
        m=market();m['ordinary'][2,0]=False
        with self.assertRaises(ValueError):simulate(m,{m['dates'][1]:(1,)},0,None)

    def test_bridge_cannot_cross_entry(self):
        m=market();m['ordinary'][2,0]=False;m['span_start'][2,0]=0
        with self.assertRaises(ValueError):simulate(m,{m['dates'][1]:(1,)},0,None)

    def test_selection_has_no_label_filter(self):
        date=pd.Timestamp('2020-01-01');entry=date+pd.Timedelta(days=1)
        pred=pd.DataFrame({'PERMNO':range(1,11),'signal_date':date,'signal_eligible':True,'label_status':'missing'})
        for model in MODELS[:-1]:pred[model]=1.
        t=make_targets(pred,pd.DataFrame({'signal_date':[date],'entry_date':[entry]}))
        self.assertEqual(t[MODELS[0]][entry],(1,));self.assertEqual(len(t['equal_weight'][entry]),10)

    def test_read_predicate_excludes_test_and_train(self):
        import tempfile
        d=pd.DataFrame({'PERMNO':[1,1,1],'signal_date':pd.to_datetime(['2019-01-01','2020-01-01','2022-01-01']),
            'split':['train','validation','test'],'signal_eligible':True,'label_end_date':pd.to_datetime(['2019-01-10','2020-01-10','2022-01-10']),'purged_boundary':False})
        for model in MODELS[:-1]:d[model]=1.
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'synthetic.parquet';d.to_parquet(p,index=False);r=read_predictions(p)
        self.assertEqual(len(r),1);self.assertEqual(r.split.iloc[0],'validation')


if __name__=='__main__':unittest.main()

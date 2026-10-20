"""Original ledger arithmetic retained only for numerical regression tests."""
import pandas as pd
from equity_research.backtest.calendar import is_settlement_day

class Ledger:
    def __init__(self,cash=1.):
        self.cash=float(cash)
        self.entries=[]
        self.history=[]

    def post(self,date,due,value,owner):
        value=float(value)
        assert pd.Timestamp(due)>pd.Timestamp(date)
        assert is_settlement_day(due)
        e=dict(date=pd.Timestamp(date),due_date=pd.Timestamp(due),value=value,PERMNO=int(owner))
        self.entries.append(e);self.history.append(e.copy())

    def settle(self,date):
        # Net every due date together, never create a false intraday debit by row order.
        dates=sorted({e['due_date'] for e in self.entries if e['due_date']<=date})
        released=0.
        for due in dates:
            amount=sum(e['value'] for e in self.entries if e['due_date']==due)
            self.cash+=amount;released+=amount
            assert self.cash>=-1e-10,'Settlement funding shortfall'
            self.entries=[e for e in self.entries if e['due_date']!=due]
        return released

    def balances(self):
        return (sum(max(e['value'],0) for e in self.entries),
                sum(max(-e['value'],0) for e in self.entries))

    def path(self):
        balance=self.cash;result=[]
        for due in sorted({e['due_date'] for e in self.entries}):
            balance+=sum(e['value'] for e in self.entries if e['due_date']==due)
            result.append((due,balance))
        return result

    def budget(self,due,settled_only=False):
        # A new payable consumes every projected balance from its due date onward.
        dates=sorted({due}|{e['due_date'] for e in self.entries})
        balance=self.cash;headroom=[]
        for day in dates:
            balance+=sum(e['value'] for e in self.entries if e['due_date']==day)
            assert balance>=-1e-10,'Existing obligations not funded'
            if day>=due:headroom.append(balance)
        amount=max(0.,min(headroom))
        if settled_only:
            amount=min(amount,max(0.,self.cash-self.balances()[1]))
        return amount

    def check(self):
        assert self.cash>=-1e-10
        assert all(b>=-1e-10 for _,b in self.path()),'Projected financing required'

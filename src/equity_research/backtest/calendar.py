"""Bind a verified calendar to the shared settlement and ledger implementations."""
from equity_research.paths import ROOT
from equity_research.backtest.settlement import SettlementCalendar
from equity_research.backtest.ledger import Ledger as BaseLedger

CONFIG = ROOT / 'config/test_execution_calendar_v1.json'
_calendar = SettlementCalendar(CONFIG)
CALENDAR = _calendar.config
supported = _calendar.supported
is_settlement_day = _calendar.is_settlement_day
settlement_date = _calendar.settlement_date
is_exchange_day = _calendar.is_exchange_day


class Ledger(BaseLedger):
    """Bind the shared accounting logic to this verified calendar."""
    def __init__(self, cash=1.):
        super().__init__(cash, is_settlement_day=is_settlement_day)

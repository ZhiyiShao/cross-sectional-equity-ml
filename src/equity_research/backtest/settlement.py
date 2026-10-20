"""Settlement arithmetic shared by validation and full-test calendars."""
import json
import pandas as pd


class SettlementCalendar:
    def __init__(self, path):
        self.config = json.loads(path.read_text(encoding="utf-8"))
        self.holidays = set(pd.to_datetime(self.config["non_settlement_weekdays"]))
        self.windows = [(pd.Timestamp(a), pd.Timestamp(b))
                        for a, b in self.config["supported_windows"]]

    def supported(self, date):
        date = pd.Timestamp(date).normalize()
        if not any(a <= date <= b for a, b in self.windows):
            raise ValueError("Unverified settlement calendar date: " + str(date.date()))
        return date

    def is_settlement_day(self, date):
        date = self.supported(date)
        return date.weekday() < 5 and date not in self.holidays

    def is_exchange_day(self, date):
        date = pd.Timestamp(date).normalize()
        lo, hi = map(pd.Timestamp, self.config["exchange_supported_window"])
        if not lo <= date <= hi:
            raise ValueError("Unverified public exchange date")
        return date.weekday() < 5 and str(date.date()) not in self.config["exchange_closed_weekdays"]

    def settlement_date(self, trade_date):
        day = self.supported(trade_date)
        lag = 3 if day < pd.Timestamp("2017-09-05") else 2 if day < pd.Timestamp("2024-05-28") else 1
        for _ in range(lag):
            day += pd.Timedelta(days=1)
            while not self.is_settlement_day(day):
                day += pd.Timedelta(days=1)
        return day

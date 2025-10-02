import re
from datetime import datetime, timedelta
import traceback
import json
import pandas as pd

class TemporalNormalizerEnglish:

    def __init__(self):

        self.months = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
            "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
            "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7,
            "August": 8, "September": 9, "October": 10,
            "November": 11, "December": 12
        }
        self.months = {k.lower(): v for k, v in self.months.items()}

        self.days = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
            "mon": 0, "tue": 1, "wed": 2,
            "thu": 3, "fri": 4, "sat": 5, "sun": 6
        }

        self.days = {k.lower(): v for k, v in self.days.items()}

    def analyze(self, time_str, reference_date=datetime.now()):

        time_str = time_str.lower().strip()

        # detect June 1st, May 3rd, June 10, September 2010
        months_str = "|".join(self.months.keys())
        pattern = rf"({months_str})\s+\d+"
        if re.match(pattern, time_str):
            month = re.search(pattern, time_str).group(1)

            month_num = self.months[month]
            day = re.search(r"\d+", time_str).group(0)
            day_num = int(day)
            if len(day) == 4:
                # format: september 2010
                return datetime(day_num, month_num)
            else:
                year = reference_date.year
                return datetime(year, month_num, day_num)

        # detect monday, friday, ...
        if time_str in self.days.keys():
            day_num = self.days[time_str]
            # return the next occurrence of this day
            today = reference_date.weekday()
            days_ahead = (day_num - today + 7) % 7
            next_date = reference_date + timedelta(days=days_ahead)
            return next_date
        
        # detect 3 weeks ago
        pattern = r'\b\d+\s+weeks?\s+ago\b'
        if re.match(pattern, time_str):
            val = int(time_str[0:time_str.find(" ")])
            date = reference_date + timedelta(weeks=-1*val)
            return date

        return None


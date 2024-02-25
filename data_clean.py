from datetime import datetime, timedelta
import pytz
import pandas as pd

class CleanData:
    def __init__(self, data, current_time, datapoints_num = 300):
        self.data = data
        self.current_time = current_time
        self.collected_num = datapoints_num
    # remove posts posted in 6 hours.
    def drop_recent(self):
        centraltime = pytz.timezone('America/Chicago')
        self.data['posted_time'] = pd.to_datetime(self.data.created_utc, unit='s').dt.tz_localize('UTC').dt.tz_convert(centraltime)

        timethreshold = self.current_time + timedelta(hours=-6)
        timeshreshold_central = timethreshold.astimezone(centraltime)

        new_data = self.data[self.data['posted_time'] < timeshreshold_central]
        return new_data

    def popularity(self, data):
        sorted_data = data.sort_values('score', ascending=False).groupby('subreddit')
        popular_condition = sorted_data.head(self.collected_num).index
        unpopular_condition = sorted_data.tail(self.collected_num).index
        data.loc[popular_condition, 'popularity'] = 'popular'
        data.loc[unpopular_condition, 'popularity'] = 'unpopular'
        return data

    # get cleaned dataframe
    def cleaned_data(self):
        new_data = self.drop_recent()
        new_data['selftext'] = new_data['selftext'].fillna(' ')
        new_data['title'] = new_data['title'].fillna(' ')
        new_data['text'] = new_data['title'] + new_data['selftext']

        data_with_label = self.popularity(new_data)
        data_with_label = data_with_label[~data_with_label['popularity'].isnull()]
        return data_with_label


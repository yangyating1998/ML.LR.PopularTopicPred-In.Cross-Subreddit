from sklearn.model_selection import train_test_split

class ModelData:
    def __init__(self, goal_type, community, data):
        self.goal_type = goal_type
        self.community = community
        self.cleaned_data = data

    def train_test_data(self):
        if self.goal_type == 'inside':
            data = self.cleaned_data[self.cleaned_data['subreddit'] == self.community]  # 'stocks' change to whichever community
            X_train, X_test, y_train, y_test = train_test_split(data['text'].values, data['popularity'].values, test_size=0.2)

        if self.goal_type == 'outside':
            X_train = self.cleaned_data[self.cleaned_data['subreddit'] == self.community]['text'].values
            X_test = self.cleaned_data[self.cleaned_data['subreddit'] != self.community]['text'].values
            y_train = self.cleaned_data[self.cleaned_data['subreddit'] == self.community]['popularity'].values
            y_test = self.cleaned_data[self.cleaned_data['subreddit'] != self.community]['popularity'].values
        return X_train, X_test, y_train, y_test
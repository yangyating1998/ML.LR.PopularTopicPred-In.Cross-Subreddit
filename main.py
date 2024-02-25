# existing yy_packages
import praw
import datetime

# created yy_packages
from yy_packages.scrap_subreddit import ScrapReddit
from yy_packages.data_clean import CleanData
from yy_packages.logistic_regression import LogisticModel
from yy_packages.model_data import ModelData

#use a developer id and secret
reddit_account = praw.Reddit(
    client_id="your_id",
    client_secret="your_secret",
    user_agent="script by u/your_name",
)

# collected subreddits most recent posts from ['StockMarket', 'stocks', 'wallstreetbets']
communities = ['StockMarket', 'stocks', 'wallstreetbets']
scrap = ScrapReddit(communities, reddit_account)
subreddits_df = scrap.scrap_subreddits(post_num=1000)

# clean data by:
# (1) removing posts posted in 6 hours
# (2) prepare text for future exam by combining title and post content in new col 'text'
# (3) add one new column indicating posts' popularity (relavtive # of likes)
current_time = datetime.datetime.now()
subreddits_df.to_csv(f'stock_sr_{current_time}.csv')

# subreddits_df = pd.read_csv('stock_sr_2024-02-24 22:39:01.807778.csv')
dataclean = CleanData(subreddits_df, current_time, labeled_datapoints=300) # collected 600 posts with labels; to change collected num see 'data_clean.py'
cleaned_data = dataclean.cleaned_data()


# Model Data Prepare
# Goal 'inside': Train and test classifier (popularity ~ text) on each community with logistic regression
# Goal 'outside': Train and run classifier (popularity ~ text) on other similar communities with logistic regression
modeled_data = ModelData(goal_type='outside', community='stocks', data=cleaned_data) # example: train model in 'stock'; test in other two communities
X_train, X_test, y_train, y_test = modeled_data.train_test_data()

# logistic regression model assessment
model = LogisticModel(X_train, X_test, y_train, y_test)
accuracy_list, report_list, features = model.train_and_test()

print(accuracy_list) # three accuracy scores (order: basic data, stemmed data, and lemmatized data)
print(report_list[0]) # classification_report with basic data
print(report_list[1]) # classification_report with stemmed data
print(report_list[2]) # classification_report with lemmatized data
print(features) # a dataframe with 20 featured word for each type of data


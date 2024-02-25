# Goal
**This project trained [logistic regression models](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) (popularity ~ text) with subreddit posts.**

- Code shows examples of three similar subreddits: [r/stocks](https://www.reddit.com/r/stocks/) | [r/StockMarket](https://www.reddit.com/r/StockMarket/) | [wallstreetbets](https://www.reddit.com/r/wallstreetbets/)

- subreddits can be easily replaced in [main code](main.py) - communities = ['StockMarket', 'stocks', 'wallstreetbets']

Trained model was used to predict:

(1) the popularity of test posts inside the same subreddit community. 

(2) the popularity of posts from other similar subreddit communities.

## Steps
### [Scrap reddit posts](yy_packages/scrap_subreddit.py)
- Before start: get [reddit authorized credentials](https://www.reddit.com/prefs/apps)
- Use authorized [reddit_account](main.py) to scrap most 1000 recent posts from each subreddit and organize them in a dataframe

### [Clean Data](yy_packages/data_clean.py)
- Drop posts posted inside 6 hours.
- Label 300 posts with highest [score](https://www.reddit.com/wiki/faq/) as popular and 300 with least [score](https://www.reddit.com/wiki/faq/) as unpopular. (score: upvotes - downvotes)
- Drop unlabeled posts.
- Combine title and content to a new column.

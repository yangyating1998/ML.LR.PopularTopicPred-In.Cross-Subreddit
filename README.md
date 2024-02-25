# Objectives
**This project trains [logistic regression models](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) (popularity ~ text) with [reddit](https://www.reddit.com/) posts.**

- Code shows examples of three similar subreddits: [r/stocks](https://www.reddit.com/r/stocks/) | [r/StockMarket](https://www.reddit.com/r/StockMarket/) | [r/wallstreetbets](https://www.reddit.com/r/wallstreetbets/)

- Subreddits can be easily replaced in [main code](main.py) - communities = ['StockMarket', 'stocks', 'wallstreetbets']

Trained model was used to predict:

(1) the popularity of test posts inside the same subreddit community. 

(2) the popularity of posts from other similar subreddit communities.

## Steps
### [Scrap reddit posts](yy_packages/scrap_subreddit.py)
- Before start: get [reddit authorized credentials](https://www.reddit.com/prefs/apps)
- Use authorized [reddit_account](main.py) to scrap n (n=1000, by default) most recent posts from each subreddit and organize them in a dataframe

### [Data Cleaning](yy_packages/data_clean.py)
- Drop posts posted within n (n=4, by default) hours.
- Label n (n=300, by default) posts with highest [score](https://www.reddit.com/wiki/faq/) as popular and n (n=300, by default) with least [score](https://www.reddit.com/wiki/faq/) as unpopular. (score: upvotes - downvotes)
- Drop unlabeled posts.
- Concat title and content as a new column named 'text'.

### [Text Data Vectorization](yy_packages/vectorize.py)


### [Model]()


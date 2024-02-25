# Input: list of subreddits
# Output: dataframe with most recent 1000 datapoints

import pandas as pd
class ScrapReddit():
    def __init__(self, communities, reddit_account):
        self.communities = communities
        self.reddit = reddit_account

    def scrap_subreddits(self):
        posts = []
        for community in self.communities:
            get_data = self.reddit.subreddit(community)
            sr_800 = get_data.new(limit=1000)
            for submission in sr_800:
                post = {}
                post['subreddit'] = community
                post['title'] = submission.title
                post['created_utc'] = submission.created_utc
                post['distinguished'] = submission.distinguished
                post['domain'] = submission.domain
                post['id'] = submission.id
                post['link_flair_text'] = submission.link_flair_text
                post['num_comments'] = submission.num_comments
                post['permalink'] = submission.permalink
                post['score'] = submission.score
                post['selftext'] = submission.selftext
                post['subreddit'] = submission.subreddit.display_name
                post['title'] = submission.title
                post['upvote_ratio'] = submission.upvote_ratio
                post['url'] = submission.url

                posts.append(post)

        df = pd.DataFrame(posts)

        return df

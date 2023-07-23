############################################
#Rating Product & Sorting Reviews in Amazon
############################################

#Business Problem
#One of the most important problems in e-commerce is the correct calculation of the points given to the products after sales.
#The solution to this problem is to provide more customer satisfaction for the e-commerce site, to make the product stand out for the
#sellers and it means a hassle-free shopping experience for buyers. Another problem is the correct ordering of the comments given to the products.
#Since misleading comments will directly affect the sale of the product, it will cause both financial loss and loss of customers.
#In the solution of these 2 basic problems, e-commerce site and sellers will increase their sales, while customers will
#complete their purchasing journey without any problems.

import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("/Users/birsenbayat/Desktop/miuul/PythonProgrammingForDataScience/Measurement_Problems/Rating Product&SortingReviewsinAmazon/amazon_review.csv")

df.head()
df.shape #(4915, 12)
df.info()
df.isnull().sum()

#Calculating Average Rating according to current comments and comparing with existing average rating
#In the shared data set, users gave points and comments to a product. We will evaluate the scores given by weighting them by date.
#It is necessary to compare the first average score with the weighted score according to the date to be obtained.

#Calculate the product's average score
df["overall"].mean() #4.587589013224822

#Calculate weighted average score by date
#day_diff in min 1, max 1064. 30, 30-90, 90-180, >180

time_mean = df.loc[df["day_diff"] <= 30, "overall"].mean() * 28/100 + \
            df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 26/100 + \
            df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 24/100 + \
            df.loc[df["day_diff"] > 180, "overall"].mean() * 22/100

#weighted average by date: 4.6987161061560725

#Compare and interpret the average of each time period in weighted scoring
df.loc[df["day_diff"] <= 30, "overall"].mean() #4.742424242424242
df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() #4.803149606299213
df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() #4.649484536082475
df.loc[df["day_diff"] > 180, "overall"].mean() #4.573373327180434

#When we look at these 3 time intervals, we see that the average of the 3-month and 6-month comments is lower.
#This may be related to increased satisfaction with the product.
#satisfaction has increased recently, the seller may have taken the complaints seriously and paid attention to these points.
#Therefore, it would be more fair in all respects to keep the rate of recent comment evaluations high.


#Specifying 20 reviews to be displayed on the product detail page for the product
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

#Calculating score_pos_neg_diff, score_average_rating and wilson_lower_bound scores and adding them to the data
def score_pos_neg_diff(up, down):
    return up-down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up+down)

score_average_rating(600, 700)

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

for col in df[["helpful_yes", "helpful_no"]]:
    df["score_pos_neg_diff"] = score_pos_neg_diff(df["helpful_yes"], df["helpful_no"])

for ind, row in df.iterrows():
    df.loc[ind, "score_average_rating"] = score_average_rating(df["helpful_yes"][ind], df["helpful_no"][ind])

for ind, row in df.iterrows():
    df.loc[ind, "wilson_lower_bound"] = wilson_lower_bound(df["helpful_yes"][ind], df["helpful_no"][ind])

df.head()
df.shape
df.drop("score_average_rating", axis=1, inplace=True)

df["wilson_lower_bound"].value_counts()
df[df["helpful_yes"] == 5]

#Identifying and sorting the first 20 comments by wilson_lower_bound
df.sort_values(by="wilson_lower_bound", ascending=False).head(20)


#The wilson_lower_bound value is calculated according to the confidence interval, it is more reliable.
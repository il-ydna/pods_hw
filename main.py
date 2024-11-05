import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math


f = open("movieDataReplicationSet.csv").readlines()
#D1 10/4
print("D1 ANSWERS")
def mean(lst):
    return sum(lst)/len(lst)


movieNames = f[0].split(',')[:400]
all_user_reviews = {}
for movie in movieNames:
    all_user_reviews[movie] = []

counter = 0
for p_ind in range(1, len(f)):
    participant_reviews = f[p_ind].split(',')
    for i in range(400):
        if participant_reviews[i] != '' and participant_reviews[i] != 'N/A':
            all_user_reviews[movieNames[i]].append(float(participant_reviews[i]))

#QUESTION 1.1: Using the mean, the lowest rated movie in this dataset is
#get means
mean_user_reviews = {}
for movie in movieNames:
    mean_user_reviews[movie] = mean(all_user_reviews[movie])
#sort them
sorted_means = sorted(mean_user_reviews.items(), key=lambda item:item[1])
print(f"q1.1: {sorted_means[0]}")

#QUESTION 1.2: The mean of means is
mean_of_means = mean([m[1] for m in sorted_means])
print(f"q1.2: {mean_of_means}")

#QUESTION 1.3: Using the median, the highest rated movie in this dataset is
#get medians
median_user_reviews = {}
for movie in movieNames:
    median_user_reviews[movie] = statistics.median(all_user_reviews[movie])
#sort them
sorted_medians = sorted(median_user_reviews.items(), key=lambda item:item[1])
print(f"q1.3: {sorted_medians[-1]}")

#QUESTION 1.4: Using the median, the lowest rated movie of these choices is
movie_pool = ['Downfall (2004)', 'Halloween (1978)', 'Harry Potter and the Chamber of Secrets (2002)', 'Battlefield Earth (2000)', 'Black Swan (2010)']
for i in range(len(movie_pool)):
    movie_pool[i] = (movie_pool[i], median_user_reviews[movie_pool[i]])
sorted_pool = sorted(movie_pool, key=lambda item: item[1])
print(f"q1.4: {sorted_pool[0]}")

#QUESTION 1.5: Using the mean, the highest rated movie in this dataset is
print(f"q1.5: {sorted_means[-1]}")

#QUESTION 1.6: The modal rating of the movie Independence Day (1996) is
indep_day_mode = statistics.mode(all_user_reviews['Independence Day (1996)'])
print(f"q1.6: {indep_day_mode}")

#D2 10/11
print("D2 ANSWERS")
from scipy import stats

def mad(l):
    m = mean(l)
    return sum([abs(m-val) for val in l])/len(l)


#QUESTION 2.1 What is the mean of the standard deviations of all movies?
#put all the reviews together into one list
every_movie_std = [statistics.stdev(m[1]) for m in all_user_reviews.items()]
mean_of_std = statistics.mean(every_movie_std)
print(f"q2.1: {mean_of_std}")

#QUESTION 2.2 What is the mean of the mean absolute deviations of all movies?
#get the mad of each movie
every_movie_mad = [mad(m[1]) for m in all_user_reviews.items()]
mean_of_mads = mean(every_movie_mad)
print(f"q2.2: {mean_of_mads}")

#QUESTION 2.3: What is the average correlation between two movies in this dataset (including self-correlations, as per prompt)?
# Compute the correlation coefficient matrix

data = np.genfromtxt('movieDataReplicationSet.csv', delimiter=',',skip_header=1)
data = data[:,:400]
# Convert the data to a pandas DataFrame
df = pd.DataFrame(data, columns=[f"Movie_{i+1}" for i in range(400)])

correlation_matrix = df.corr()
print(correlation_matrix.values.mean())

#QUESTION 2.4: What is the median of the mean absolute deviations of all movies?
every_movie_mad = [mad(m[1]) for m in all_user_reviews.items()]
median_of_mads = statistics.median(every_movie_mad)
print(f"q2.4: {median_of_mads}")

#QUESTION 2.5:
med_of_std = statistics.median(every_movie_std)
print(f"q2.5: {med_of_std}")

#D3 10/18
print("D3 ANSWERS")
#3.1 how many joint raters of Star wars 1 and 2?
sw1 = 'Star Wars: Episode 1 - The Phantom Menace (1999)'
sw2 = 'Star Wars: Episode II - Attack of the Clones (2002)'
tit = 'Titanic (1997)'
data = np.genfromtxt('movieDataReplicationSet.csv',delimiter=',')
df = pd.DataFrame(data[1:,:400], columns=movieNames)
sw_ratings_overlap = df[sw1].notna() & df[sw2].notna()
sw_overlaps_count = sw_ratings_overlap.sum()
print(f"3.1: {sw_overlaps_count}")

#3.2 beta of sw1 and tit
#trimming the df so it only has review of both sw1 and sw2

sw1_tit_ratings_overlap = df[sw1].notna() & df[tit].notna()
sw1_tit_df = df[sw1_tit_ratings_overlap][[sw1, tit]]


def beta(x, y):
    if(len(x) != len(y)):
        return
    x_bar = x.mean()
    y_bar = y.mean()
    numer = float((x - x_bar).dot(y - y_bar))
    denom = float((x - x_bar).dot(x - y_bar))
    return numer/denom


sw1_tit_b=beta(sw1_tit_df[sw1], sw1_tit_df[tit])
print(f"3.2: {sw1_tit_b}")

#3.3
print(f"3.3: {sum(sw1_tit_ratings_overlap)}")

#3.4
sw1_sw2_df = df[sw_ratings_overlap][[sw1, sw2]]
sw1_sw2_b = beta(sw1_sw2_df[sw2], sw1_sw2_df[sw1])
print(f"3.4: {sw1_sw2_b}")

#3.5
sw1_sw2_tit_ratings_overlap = df[sw1].notna() & df[tit].notna() & df[sw2].notna()
print(f"3.5: {sum(sw1_sw2_tit_ratings_overlap)}")

#HOMEWORK 4
print('HOMEWORK 4')

income_data = pd.read_csv('movieDataReplicationSet.csv')
income_data = income_data[['income', 'Education', 'SES']].to_numpy()
inc = income_data[:, 0]
edu = income_data[:, 1]
ses = income_data[:, 2].reshape(len(income_data), 1)

#4.1 correlation between edu and income
print(f"q4.1: {np.corrcoef(income_data[:, 0], income_data[:, 1])[0][1]}")

#multiple corr
edu_and_ses = income_data[:,1:]
multipleReg = LinearRegression().fit(edu_and_ses, inc)
#4.2 get the coef of education to income
print(f"4.2: {multipleReg.coef_[0]}")

#4.3 get the rmse of the multiple regression
#getting y-hat
mult_reg_yhat = (multipleReg.coef_[0] * edu).flatten() + (multipleReg.coef_[1] * ses).flatten() + multipleReg.intercept_
#calcing resids
mult_reg_resid = inc - mult_reg_yhat
#oh baby std time
print(f"4.3: {np.std(mult_reg_resid)}")

#4.4 corr between actual/predicted outcomes
print(f"4.4: {np.corrcoef(inc, mult_reg_yhat)[0][1]}")

#4.5: COD (aka R^2) of the mult reg
print(f"4.5: {multipleReg.score(edu_and_ses, inc)}")

#4.6, partial correlation
ses_to_inc_model = LinearRegression().fit(ses, inc)
ses_to_inc_yhat = ses_to_inc_model.coef_ * ses + ses_to_inc_model.intercept_
ses_to_inc_resid = inc - ses_to_inc_yhat.flatten()

ses_to_edu_model = LinearRegression().fit(ses, edu)
ses_to_edu_yhat = ses_to_edu_model.coef_ * ses + ses_to_edu_model.intercept_
ses_to_edu_resid = edu - ses_to_edu_yhat.flatten()

partialCorr = np.corrcoef(ses_to_inc_resid, ses_to_edu_resid)
print(f"4.6: {partialCorr[0][1]}")


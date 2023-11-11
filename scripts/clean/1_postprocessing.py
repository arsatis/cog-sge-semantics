import csv
import re
import numpy as np
import pandas as pd
from collections import defaultdict


data = []
title = None


def read_csv():
    """
    Reads anonymized data from `data_anonymized.csv`, which is not available due to privacy concerns.
    """
    global title
    
    with open('data/data_anonymized.csv', 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        title = next(csv_reader)
        for line in csv_reader:
            tmp = {title[i]: line[i] for i in range(len(line))}
            data.append(tmp)
    print('Total # participants:', len(data), '\n')

    # remove non-essential fields
    n = len(title)
    ne_fields = {
        'Status', 'IPAddress', 'ResponseId', 'RecipientLastName', 'RecipientFirstName', 'RecipientEmail',
        'ExternalReference', 'LocationLatitude', 'LocationLongitude', 'DistributionChannel',
        'Demographics', 'Q1', 'Q2', 'Q3', 'Q4', 'Q4_8_TEXT', 'Q5', 'Q5_4_TEXT', 'Q6', 'Q6_4_TEXT',
        'Q7', 'Q8', 'Q9'
    }
    for i in range(n - 1, -1, -1):
        if title[i] in ne_fields:
            title.pop(i)


def write_csv():
    """
    Writes cleaned data to `data_cleaned.csv`, which is available under `data/`.
    """
    with open('data/data_cleaned.csv', 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(title)
        for d in data:
            line = [d[x] for x in title if x in d]
            csv_writer.writerow(line)


def clean_data():
    """
    Performs post-processing on the data. Specifically, this does the following:
    1. Remove non-native SgE speakers
    2. Remove participants by number of years spent in Singapore (optional)
    3. Remove participants who failed some attention check question
    4. Remove participants based on the procedure in SimLex-999 (Hill et al., 2015)
    5. Remove outliers in terms of time taken to complete the study
    """
    enable_removal_by_years_lived = False
    enable_stricter_attention_checks = True
    enable_score_adjustment = True

    global data
    to_remove = set()

    # remove participants whose native language is not English
    print('--- non-Singaporeans ---')
    for i in range(len(data) - 1, -1, -1):
        if data[i]['Q4'] != 'English':
            to_remove.add(i)
    print('# participants whose native language is not English (or who left this column blank):', len(to_remove))

    # remove participants who have lived in singapore for < 10 years (i.e., not exactly a 'native Singapore English speaker')
    if enable_removal_by_years_lived:
        for i in range(len(data) - 1, -1, -1):
            if data[i]['Q8'] == '' or int(data[i]['Q8']) < 10:
                data.pop(i)
        print('# participants who lived in Singapore for < 10 years:', len(to_remove))
    print('# participants after removing incomplete responses and non-native SgE speakers:', len(data) - len(to_remove), '\n')

    # remove participants who failed any of the attention checks
    num_failed = defaultdict(int)
    criteria = {}
    checks = {
        'Q3_12': ['select 2', 2, 'must select 2'],
        'Q6_12': ['snake ~ snake', 6, 'must select 6'],
        'Q7_12': ['apple ~ apple', 6, 'must select 6'],
        'Q9_12': ['select 5', 5, 'must select 5'],
    }
    for i in range(len(data) - 1, -1, -1):
        for k, v in checks.items():
            if int(data[i][k]) != v[1]:
                to_remove.add(i)
                num_failed[v[0]] += 1
                criteria[v[0]] = v[2]
                break

    # stricter attention checks, may not be necessary if it filters out too many participants
    if enable_stricter_attention_checks:
        stricter_checks = {
            'Q1_12': ['nose ~ house', 3, 'must select < 3'],
            'Q2_12': ['sky ~ angry', 3, 'must select < 3'],
            'Q4_12': ['grass ~ glass', 3, 'must select < 3'],
            'Q5_12': ['sound ~ on', 3, 'must select < 3'],
            'Q8_12': ['hair ~ sight', 3, 'must select < 3'],
            'Q10_12': ['moon ~ wood', 3, 'must select < 3'],
            'Q11_12': ['unemployment ~ cat', 3, 'must select < 3'],
        }
        for i in range(len(data) - 1, -1, -1):
            for k, v in stricter_checks.items():
                if int(data[i][k]) >= v[1]:
                    to_remove.add(i)
                    num_failed[v[0]] += 1
                    criteria[v[0]] = v[2]
                    break

    # increase/decrease the rating of respondents whose mean rating is greater than 1 from the mean across all participants
    # for each pair by one, except in cases where they had given the maximum/minimum rating
    if enable_score_adjustment:
        resp_means = []
        for d in data:
            d = [int(v) for k, v in d.items() if re.search('^Q[0-9]+_[0-9]+$', k)] # extract responses to the survey questions
            resp_means.append(sum(d) / len(d))
        np_resp_means = np.array(resp_means)
        mean = np.mean(np_resp_means, axis=0)

        num_deviations = 0
        for i in range(len(resp_means)):
            if mean - resp_means[i] > 1:
                for k in data[i]:
                    if re.search('^Q[0-9]+_[0-9]+$', k) and data[i][k] not in {'0', '6'}:
                        data[i][k] = str(int(data[i][k]) + 1)
                num_deviations += 1
            elif mean - resp_means[i] < -1:
                for k in data[i]:
                    if re.search('^Q[0-9]+_[0-9]+$', k) and data[i][k] not in {'0', '6'}:
                        data[i][k] = str(int(data[i][k]) - 1)
                num_deviations += 1

    # remove participants whose average pairwise Spearman correlation of responses with all other responses
    # is > 1 SD below the mean of all such averages
    to_remove_means = set()
    resp = defaultdict(list)
    for i in range(len(data)):
        resp[i] = [int(v) for k, v in data[i].items() if re.search('^Q[0-9]+_[0-9]+$', k)]
    corr_mat = pd.DataFrame(data=resp).corr(method='spearman')
    pairwise_corr = [[corr_mat[i][j] for j in corr_mat[i].index if i != j] for i in corr_mat.index]
    mean_corr = np.array([sum(x) / len(x) for x in pairwise_corr])
    mean, sd = np.mean(mean_corr, axis=0), np.std(mean_corr, axis=0)
    for i in range(len(pairwise_corr)):
        if mean - mean_corr[i] > sd:
            to_remove_means.add(i)
    to_remove = to_remove.union(to_remove_means)
    print('--- pairwise correlations + inter-rater agreement + attention check failures ---')
    print('# participants to remove based on mean pairwise response correlations:', len(to_remove_means))

    # the increase in inter-rater agreement when a rater was excluded from the analysis needed to be smaller
    # than at least 50 other raters
    to_remove_ira = set()
    ira_without_idx = []
    for i in range(len(data)):
        ratings = []
        for j in range(len(data)):
            if j != i:
                ratings += [(k, v) for k, v in data[j].items() if re.search('^Q[0-9]+_[0-9]+$', k)]
        ira_without_idx.append((fleiss_kappa(ratings, len(data) - 1), i))
    ira_without_idx.sort()

    last_idx = len(data) // 10
    eleventh_percentile_kappa = ira_without_idx[last_idx][0]
    for i in range(last_idx):
        kappa, ps = ira_without_idx[i]
        if eleventh_percentile_kappa - kappa > 0.0001: # note: 0.0001 is just some arbitrary hyperparameter
            to_remove_ira.add(ps)
    to_remove = to_remove.union(to_remove_ira)
    print('# participants to remove based on inter-rater agreement:', len(to_remove_ira))

    # remove flagged participants
    for k, v in num_failed.items():
        print(f'# participants failing {k}: {v} (criteria: {criteria[k]})')
    to_remove = sorted(list(to_remove), reverse=True)
    for idx in to_remove: data.pop(idx)
    print('# participants after removing those failing attention checks + outliers in terms of mean pairwise response correlations:', len(data), '\n')

    # remove participants whose time taken is > 1 SD beyond the mean
    time_taken = np.array([int(x['Duration (in seconds)']) for x in data])
    mean, sd = np.mean(time_taken, axis=0), np.std(time_taken, axis=0)
    data = [x for x in data if abs(int(x['Duration (in seconds)']) - mean) <= sd]
    print('--- other checks ---')
    print('# participants after removing outliers in terms of time taken (1 SD beyond mean):', len(data))


# Adapted from https://gist.github.com/ShinNoNoir/4749548
def fleiss_kappa(ratings, n):
    '''
    Computes the Fleiss' kappa measure for assessing the reliability of 
    agreement between a fixed number n of raters when assigning categorical
    ratings to a number of items.
    
    Args:
        ratings: a list of (item, category)-ratings
        n: number of raters
    Returns:
        the Fleiss' kappa score
    
    See also: http://en.wikipedia.org/wiki/Fleiss'_kappa
    '''
    items = set()
    categories = set()
    n_ij = {}
    
    for i, c in ratings:
        items.add(i)
        categories.add(c)
        n_ij[(i,c)] = n_ij.get((i,c), 0) + 1

    N = len(items)
    P_i = dict(((i, (sum(n_ij.get((i, c), 0) ** 2 for c in categories) - n) / (n * (n - 1.0))) for i in items))
    p_j = dict(((c, sum(n_ij.get((i, c), 0) for i in items) / (1.0 * n * N)) for c in categories))

    P_bar = sum(P_i.values()) / (1.0 * N)
    P_e_bar = sum(value ** 2 for value in p_j.values())
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return kappa


if __name__ == '__main__':
    read_csv()
    clean_data()
    write_csv()

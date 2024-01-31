import pandas as pd
import numpy as np
import json
from scipy.stats._stats import _kendall_dis


def calculate_somersd(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size:
        raise ValueError("All inputs must be of the same size, "
                         "found x-size %s and y-size %s" % (x.size, y.size))
    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        return ((cnt * (cnt - 1) // 2).sum(),
            (cnt * (cnt - 1.) * (cnt - 2)).sum(),
            (cnt * (cnt - 1.) * (2*cnt + 5)).sum())
    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)
    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)
    dis = _kendall_dis(x, y)  # discordant pairs
    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.where(obs)[0]).astype('int64', copy=False)
    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1 = count_rank_tie(x)     # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)     # ties in y, stats
    tot = (size * (size - 1)) // 2
    SD = (tot - xtie - ytie + ntie - 2 * dis) / (tot - ntie)
    return (SD, dis)


if __name__ == '__main__':
    generated_dataset = pd.read_csv(r'C:\Users\DE129454\Documents\Datengenerierung\chatgbt_generierung_2.csv', sep=",")
    wanted_correlation = pd.read_csv(r'C:\Users\DE129454\Documents\Datengenerierung\Correlationmatrix.csv', sep=";", decimal=",")
    wanted_correlation.set_index('Unnamed: 0', inplace=True)
    wanted_correlation.index.name = None
    wanted_correlation = wanted_correlation.astype(float)
    wanted_correlation = wanted_correlation.fillna(0.0)
    wanted_correlation = wanted_correlation + wanted_correlation.T
    np.fill_diagonal(wanted_correlation.values, 1.0)
    new_correlation = generated_dataset.corr()
    corr_difference = new_correlation.subtract(wanted_correlation)
    corr_difference = corr_difference.abs()
    csv_dataset = corr_difference.to_csv(r'C:\Users\DE129454\Documents\Datengenerierung\correlation_difference.csv', sep=';', decimal=',')
    json_file = {}
    for col in generated_dataset.columns:
        median = str(generated_dataset[col].median())
        min = str(generated_dataset[col].min())
        max = str(generated_dataset[col].max())
        q25 = str(generated_dataset[col].quantile(0.25))
        q75 = str(generated_dataset[col].quantile(0.75))
        if col == "Default Flag":
            json_file[col] = {"median": median, "min": min, "max": max}
        else:
            (somersd_calc, dist_calc) = calculate_somersd(generated_dataset[col], generated_dataset["Default Flag"])
            json_file[col] = {"median": median, "min": min, "max": max, "q25": q25, "q75": q75, "somers'd": str(somersd_calc)}
    
    with open(r'C:\Users\DE129454\Documents\Datengenerierung\statistics_of_generated.json', "w") as write_file:
        json.dump(json_file, write_file, indent=4)

    print(generated_dataset.head())
    print(wanted_correlation.head())
    print(new_correlation.head())
    print(corr_difference)

import pandas as pd
import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats._stats import _kendall_dis
from fitter import Fitter, get_common_distributions, get_distributions


# read config of wanted statistics
def read_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    f.close()
    parameters = config["variables"]
    variables = []
    for variable in parameters:
        variables.append(variable)
    return parameters, variables


# manual computation of somersd between risk driver (x) and binary label (y)
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


# check data column for different distributions and return the most probable
def get_distribution(data_array):
    f = Fitter(data_array, distributions=["alpha", "beta", "gamma", "logistic", "uniform", "norm", "expon"]) # distribution to check
    f.fit()
    best_json = f.get_best(method= 'sumsquare_error')
    best_array = []
    for best in best_json:
        best_array.append(best)
    return best_array[0]


 # All possible distributions
 # ['alpha', 'anglit', 'arcsine', 'argus', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy',
 # 'chi', 'chi2', 'cosine', 'crystalball', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponpow',
 # 'exponweib', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'gamma', 'gausshyper', 'genexpon',
 # 'genextreme', 'gengamma', 'genhalflogistic', 'genhyperbolic', 'geninvgauss', 'genlogistic', 'gennorm',
 # 'genpareto', 'gilbrat', 'gompertz', 'gumbel_l', 'gumbel_r', 'halfcauchy', 'halfgennorm', 'halflogistic',
 # 'halfnorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa3',
 # 'kappa4', 'ksone', 'kstwo', 'kstwobign', 'laplace', 'laplace_asymmetric', 'levy', 'levy_l', 'levy_stable',
 # 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'loguniform', 'lomax', 'maxwell', 'mielke', 'moyal',
 # 'nakagami', 'ncf', 'nct', 'ncx2', 'norm', 'norminvgauss', 'pareto', 'pearson3', 'powerlaw',
 # 'powerlognorm', 'powernorm', 'rayleigh', 'rdist', 'recipinvgauss', 'reciprocal', 'rice', 'rv_continuous',
 # 'rv_histogram', 'semicircular', 'skewcauchy', 'skewnorm', 'studentized_range', 't', 'trapezoid', 'trapz',
 # 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald',
 # 'weibull_max', 'weibull_min', 'wrapcauchy']


def main(input_dir_path, output_dir_path, dataset):
    #print(get_distributions()) # distributions supported by scipy
    generated_dataset = pd.read_csv(os.path.join(input_dir_path, dataset), sep=",")
    wanted_correlation = pd.read_csv(os.path.join(input_dir_path, 'Correlationmatrix.csv'), sep=";", decimal=",")
    parameters, variables = read_config(os.path.join(input_dir_path, 'Statistische_Informationen.json'))
    wanted_correlation.set_index('Unnamed: 0', inplace=True)
    wanted_correlation.index.name = None
    wanted_correlation = wanted_correlation.astype(float)
    wanted_correlation = wanted_correlation.fillna(0.0)
    wanted_correlation = wanted_correlation + wanted_correlation.T
    np.fill_diagonal(wanted_correlation.values, 1.0)
    new_correlation = generated_dataset.corr()
    corr_difference = new_correlation.subtract(wanted_correlation)
    corr_difference = corr_difference.abs()
    # correlation difference (difference between wanted corr and achieved corr) csv
    csv_dataset = corr_difference.to_csv(os.path.join(output_dir_path, 'correlation_difference.csv'), sep=';', decimal=',')
    # correlation difference heatmap
    plt.figure()
    ax = sns.heatmap(corr_difference, annot=True, vmin=0, vmax=1)
    plt.savefig(os.path.join(output_dir_path, 'correlation_difference.png'), bbox_inches = "tight")
    somersd_value = []
    somersd_type = []
    labels = []
    json_file = {}
    # computation of all statistical information and differences with the wanted one
    for col in generated_dataset.columns:
        distribution = get_distribution(generated_dataset[col].values)
        median = round(generated_dataset[col].median(), 3)
        min = round(generated_dataset[col].min(), 3)
        max = round(generated_dataset[col].max(), 3)
        q25 = round(generated_dataset[col].quantile(0.25), 3)
        q75 = round(generated_dataset[col].quantile(0.75), 3)
        median_should = parameters[col]["median"]
        min_should = parameters[col]["min"]
        max_should = parameters[col]["max"]
        median_dif = round(abs(median - float(median_should)), 3)
        min_dif = round(abs(min - float(min_should)), 3)
        max_dif = round(abs(max - float(max_should)), 3)
        param_names = []
        for param in parameters[col]:
            param_names.append(param)
        if col == "Default Flag":
            json_file[col] = {"median": str(median), "median_should": median_should, "median_difference": str(median_dif), "min": str(min), "min_should": min_should, "min_difference": str(min_dif), "max": str(max), "max_should": max_should, "max_difference": str(max_dif), "distribution": distribution}
        elif "q25" not in param_names:
            (somersd_calc, dist_calc) = calculate_somersd(generated_dataset[col], generated_dataset["Default Flag"])
            somersd_calc = round(somersd_calc, 3)
            somersd_should = parameters[col]["somers'd"]
            somersd_dif = round(abs(somersd_calc - float(somersd_should)), 3)
            somersd_value.append(somersd_calc)
            somersd_type.append('actual')
            labels.append(col)
            somersd_value.append(float(somersd_should))
            somersd_type.append('target')
            labels.append(col)
            somersd_value.append(somersd_dif)
            somersd_type.append('difference')
            labels.append(col)
            json_file[col] = {"median": str(median), "median_should": median_should, "median_difference": str(median_dif), "min": str(min), "min_should": min_should, "min_difference": str(min_dif), "max": str(max), "max_should": max_should, "max_difference": str(max_dif), "somers'd": str(somersd_calc), "somers'd_should": somersd_should, "somers'd_difference": str(somersd_dif), "distribution": distribution}
        else:
            q25_should = parameters[col]["q25"]
            q75_should = parameters[col]["q75"]
            q25_dif = round(abs(q25 - float(q25_should)), 3)
            q75_dif = round(abs(q75 - float(q75_should)), 3)
            (somersd_calc, dist_calc) = calculate_somersd(generated_dataset[col], generated_dataset["Default Flag"])
            somersd_calc = round(somersd_calc, 3)
            somersd_should = parameters[col]["somers'd"]
            somersd_dif = round(abs(somersd_calc - float(somersd_should)), 3)
            somersd_value.append(somersd_calc)
            somersd_type.append('actual')
            labels.append(col)
            somersd_value.append(float(somersd_should))
            somersd_type.append('target')
            labels.append(col)
            somersd_value.append(somersd_dif)
            somersd_type.append('difference')
            labels.append(col)
            json_file[col] = {"median": str(median), "median_should": median_should, "median_difference": str(median_dif), "min": str(min), "min_should": min_should, "min_difference": str(min_dif), "max": str(max), "max_should": max_should, "max_difference": str(max_dif), "q25": str(q25), "q25_should": q25_should, "q25_difference": str(q25_dif), "q75": str(q75), "q75_should": q75_should, "q75_difference": str(q75_dif), "somers'd": str(somersd_calc), "somers'd_should": somersd_should, "somers'd_difference": str(somersd_dif), "distribution": distribution}
    data = pd.DataFrame({'labels': labels, 'value': somersd_value, 'type': somersd_type})
    # somers'd bar plot
    plt.figure()
    ax = sns.barplot(data=data, x='labels', y='value', hue='type', hue_order=['actual', 'target', 'difference'])
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output_dir_path, 'somers_d.png'), bbox_inches = "tight")
    # json of data statistics
    with open(os.path.join(output_dir_path, 'statistics_of_generated.json'), "w") as write_file:
        json.dump(json_file, write_file, indent=4)


if __name__ == '__main__':
    input_dir_path = r'C:\Users\DE129454\Documents\Datengenerierung'
    output_dir_path = r'C:\Users\DE129454\Documents\Datengenerierung'
    dataset = 'chatgbt_generierung_2.csv'
    main(input_dir_path, output_dir_path, dataset)

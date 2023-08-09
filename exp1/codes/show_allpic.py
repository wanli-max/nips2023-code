import csv
import numpy as np
import json
import matplotlib.pyplot as plt
from numpy.lib.type_check import real
import seaborn as sns
import os

sns.set(style="darkgrid")


def getData(path):
    rawData = open(path, encoding="utf-8")
    rawData = json.load(rawData)
    rawData = np.array(rawData)
    return rawData[0], rawData[1]

def getFilePath_list(algo_name):
    data_path ="xxx"+algo_name
    fileList = os.listdir(data_path)
    x = []
    for file in fileList:
        x.append(os.path.join(data_path, file))
    return x, fileList


def smooth(data, weight=0.8):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def get_data_stat(filepath_list):
    data_space = 1
    training_iters = 60

    x = np.arange(1, training_iters+1, data_space)
    data_length = int(training_iters/data_space)
    
    data_list = []
    for file_path in filepath_list:

        _, data = getData(file_path)
        np_data = data[0:60].reshape(data_length, 1)
        data_list.append(np_data)
    data_list = np.concatenate(data_list, 1)

    return x, np.mean(data_list, 1), np.std(data_list, 1)


alg_lists = ["33q_de", "33q_ind", "q_de", "q_ind"]
color_map = {
    "33q_de": sns.xkcd_rgb["kelly green"],
    "33q_ind": sns.xkcd_rgb["orange"],
    "q_de": sns.xkcd_rgb["red"],
    "q_ind": sns.xkcd_rgb["slate blue"]
}

# alg_lists = ["pomdp_ind_3", "pomdp_de_3"]
# color_map = {
#     "fullob_ind_7": sns.xkcd_rgb["kelly green"],
#     "pomdp_de_3": sns.xkcd_rgb["red"],
#     "pomdsp_ind_3": sns.xkcd_rgb["orange"],
#     "fullob_de_ep0.5": sns.xkcd_rgb["brown"],
#     "pomdp_ind_3": sns.xkcd_rgb["slate blue"]
# }

fig, ax = plt.subplots(1, 1, figsize=(20, 20), dpi=100)

std_num = 0.5
# for i in range(len(ax)):
legend_list = []
for alg_name in alg_lists:
    filepath_list, _ = getFilePath_list(alg_name)
    x, mean_data, std_data = get_data_stat(filepath_list)
    mean_data = smooth(mean_data, 0.6)
    plot = ax.plot(x, mean_data, c=color_map[alg_name])
    low_std = mean_data - std_data * std_num
    high_std = mean_data + std_data * std_num
    ax.fill_between(x, low_std, high_std, alpha=0.3,
                        facecolor=color_map[alg_name])

# ax.set_xticks([0, 200, 400, 600, 800, 1000])
# ax.set_xticklabels(["0", "2k", '4k', "6k", "8k", "10k"])
for line in ax.get_lines():
    line.set_linewidth(2.0)

# ax.set_yticks([-9, -6, -3, 0, 3, 6, 9])
# ax.set_yticklabels(["-9", "-6", '-3', "0", "3", "6", "9"])

ax.set_ylim(-0.2, 4.2)

ax.legend(legend_list, labels=alg_lists, bbox_to_anchor=(-0.45, 1),
                    loc="upper left", borderaxespad=0.1, )

plt.show()
plt.savefig("first3.png")

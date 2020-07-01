import json
from statistics import mean

import os

from helpers.constants import osvos_save_file_path, osvos_save_file_path_all_results
from osvos.train_osvos import main as train_osvos


def main():
    imagesets_path = os.path.join('data/DAVIS_2016', 'ImageSets', '480p')

    with open(os.path.join(imagesets_path, f'val.txt'), 'r') as f:
        tmp = f.readlines()
    sequences_names = sorted({x.split()[0].split('/')[-2] for x in tmp})

    all_results_model = {}
    for seq in sequences_names:
        all_results = train_osvos(sequence_name=seq)
        all_results_model[seq] = all_results
        with open(osvos_save_file_path_all_results, 'w') as f:
            json.dump(all_results_model, f)

    jf_means = []
    j_means = []
    f_means = []
    eval_times = []
    for key in all_results_model:
        idx = len(all_results_model[key]) - 2
        score = all_results_model[key][idx]
        jf_means.append(score['jfmean'])
        j_means.append(score['jmean'])
        f_means.append(score['fmean'])
        eval_times.append(score['eval_time'])

    jf_mean = mean(jf_means)
    j_mean = mean(j_means)
    f_mean = mean(f_means)
    total_time = mean(eval_times)
    with open(osvos_save_file_path, 'w') as f:
        # Print the results
        print(f"--------------------------- Global results for val ---------------------------\n", file=f)
        print(f'JF-Mean: {jf_mean}  F-Mean: {f_mean}    J-Mean: {j_mean}, total_time: {total_time}', file=f)


if __name__ == '__main__':
    main()

import os
from osvos.train_osvos import main as train_osvos
from collections import defaultdict
from helpers.constants import model_name
import numpy as np
import json


def main():
    imagesets_path = os.path.join('data/DAVIS_2016', 'ImageSets', '480p')

    with open(os.path.join(imagesets_path, f'val.txt'), 'r') as f:
        tmp = f.readlines()
    sequences_names = sorted({x.split()[0].split('/')[-2] for x in tmp})

    results = defaultdict(list)
    all_results_model = {}
    save_file_path = f'osvos/results/{model_name}_osvos.txt'
    save_file_path_all_results = f'osvos/results/{model_name}_osvos_all_results.json'
    for seq in sequences_names:
        best_f_mean, best_j_mean, total_time, beginning_jf_mean, all_results = train_osvos(sequence_name=seq)
        results['FMean'].append(best_f_mean)
        results['JMean'].append(best_j_mean)
        results['Time'].append(total_time)
        results['Start-JF'].append(beginning_jf_mean)
        all_results_model[seq] = all_results
        with open(save_file_path_all_results, 'w') as f:
            json.dump(all_results_model, f)

        j_mean = np.mean(results['FMean'])
        f_mean = np.mean(results['JMean'])
        jf_mean = (j_mean + f_mean) / 2
        total_time = np.sum(results['Time'])
        with open(save_file_path, 'w') as f:
            # Print the results
            print(f"--------------------------- Global results for val ---------------------------\n", file=f)
            print(f'JF-Mean: {jf_mean}  F-Mean: {f_mean}    J-Mean: {j_mean}, total_time: {total_time}', file=f)
            print(f"\n---------- Per sequence results for val ----------\n", file=f)
            print("Sequence    J-Mean    F-Mean     Time    Start-JFMean", file=f)
            for seq_name, fmean, jmean, time, start_jf in zip(sequences_names, results['FMean'], results['JMean'], results['Time'],
                                                              results['Start-JF']):
                print(f"{seq_name}  {jmean:.6f}    {fmean:.6f}    {time:.6f}    {start_jf}", file=f)


if __name__ == '__main__':
    main()

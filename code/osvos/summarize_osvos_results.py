import json
from statistics import mean

if __name__ == '__main__':
    json_path = 'data/output/osvos_results/model_maskrcnn_slowfast_sp_1fp_1_all_results.json'

    with open(json_path) as f:
        all_results_model = json.load(f)

    for epoch_idx in range(10):
        jf_means = []
        j_means = []
        f_means = []
        eval_times = []
        for key in all_results_model:
            idx = str(epoch_idx)
            score = all_results_model[key][idx]
            jf_means.append(score['jfmean'])
            j_means.append(score['jmean'])
            f_means.append(score['fmean'])
            eval_times.append(score['eval_time'])

        jf_mean = mean(jf_means)
        j_mean = mean(j_means)
        f_mean = mean(f_means)
        total_time = mean(eval_times)

        print(f'{epoch_idx}-JF:{jf_mean:.3f}-J:{j_mean:.3f}-F:{f_mean:.3f}-Time:{total_time:.3f}')

'''
0-JF:0.697-J:0.676-F:0.718-Time:63.127
1-JF:0.701-J:0.680-F:0.722-Time:62.568
2-JF:0.706-J:0.685-F:0.727-Time:62.664
3-JF:0.703-J:0.682-F:0.724-Time:62.911
4-JF:0.712-J:0.691-F:0.732-Time:63.071

'''

import json
from statistics import mean

from helpers.constants import slow_pathway_size, fast_pathway_size, osvos_experiments_path
from osvos.experiment_config import ExperimentConfig
from osvos.train_osvos import main as train_osvos


def main():
    freeze_options = ['none', 'SF', 'BB_SF']
    scales = [0.25, 0.4]
    lrs = [0.001, 0.0005, 0.0001, 0.005]
    sequences_names = ['breakdance', 'bmx-trees']

    current_idx = 0
    total = len(freeze_options) * len(scales) * len(lrs)
    for freeze in freeze_options:
        for scale in scales:
            for lr in lrs:
                config = ExperimentConfig(freeze=freeze, lr=lr, scale=scale, epochs=5)

                config_name = f"osvos_sp_{slow_pathway_size}fp_{fast_pathway_size}_freeze_{freeze}_scale_{scale}_lr_{lr}"
                json_output_path = osvos_experiments_path / f"{config_name}.json"
                txt_output_path = osvos_experiments_path / f"{config_name}.txt"

                # skip experiments that were already done
                if json_output_path.exists():
                    print(f"Skipping {current_idx} / {total}: {str(config)}.")
                    current_idx += 1
                    continue
                print(f"{current_idx} / {total}: {str(config)}")
                current_idx += 1

                all_results_model = {}
                for seq in sequences_names:
                    all_results = train_osvos(sequence_name=seq, cfg=config)
                    all_results_model[seq] = all_results
                    with open(json_output_path, 'w') as f:
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
                with open(txt_output_path, 'w') as f:
                    # Print the results
                    print(f"--------------------------- Global results for val ---------------------------\n", file=f)
                    print(f'JF-Mean: {jf_mean}  F-Mean: {f_mean}    J-Mean: {j_mean}, total_time: {total_time}', file=f)


if __name__ == '__main__':
    main()

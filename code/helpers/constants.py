from pathlib import Path

import shutil

environment = 'colab'  # or colab
print(f'Environment is {environment}')
slow_pathway_size = 7
fast_pathway_size = 7
continue_training = False
use_caching = False
model_name = f'model_maskrcnn_slowfast_sp_{slow_pathway_size}fp_{fast_pathway_size}'
random_seed = 63
root_dir_path = Path('/content/gdrive/My Drive/Python Projects/adl4cv_root') if environment == 'colab' else Path('')
writer_dir = (root_dir_path / 'runs') / model_name
if writer_dir.exists():
    shutil.rmtree(writer_dir)
root_dir_path.mkdir(parents=True, exist_ok=True)
models_dir_path = root_dir_path / Path('models')
models_dir_path.mkdir(parents=True, exist_ok=True)
best_model_path = models_dir_path / (f'{model_name}_best.pth')
model_path = models_dir_path / (f'{model_name}.pth')
checkpoint_path = models_dir_path / (f'{model_name}_checkpoint.pth')
data_output_path = root_dir_path / Path('data/output')
data_output_path.mkdir(parents=True, exist_ok=True)
eval_output_path = (data_output_path / 'eval_output')
eval_output_path.mkdir(parents=True, exist_ok=True)
pred_output_path = (data_output_path / 'pred_output')
pred_output_path.mkdir(parents=True, exist_ok=True)
osvos_save_root_path = data_output_path / f'osvos_results'
osvos_save_root_path.mkdir(parents=True, exist_ok=True)
osvos_save_file_path = osvos_save_root_path / f'{model_name}.txt'
osvos_save_file_path_all_results = osvos_save_root_path / f'{model_name}_all_results.json'

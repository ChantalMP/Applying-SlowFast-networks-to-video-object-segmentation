from pathlib import Path

batch_size = 2
maskrcnn_batch_size = 8  # 4 and 16 for google colab
slow_pathway_size = 4
fast_pathway_size = 4
model_name = f'model_maskrcnn_slowfast_sp_{slow_pathway_size}fp_{fast_pathway_size}'
root_dir_path = Path('')  # For colab '/gdrive/My\ Drive/Python\ Projects/adl4cv_root'
root_dir_path.mkdir(parents=True, exist_ok=True)
models_dir_path = root_dir_path / Path('models')
models_dir_path.mkdir(parents=True, exist_ok=True)
best_model_path = models_dir_path / (f'{model_name}_best.pth')
model_path = models_dir_path / (f'{model_name}.pth')
data_output_path = root_dir_path / Path('data/output')
data_output_path.mkdir(parents=True, exist_ok=True)
eval_output_path = (data_output_path / 'eval_output')
eval_output_path.mkdir(parents=True, exist_ok=True)
pred_output_path = (data_output_path / 'pred_output')
pred_output_path.mkdir(parents=True, exist_ok=True)

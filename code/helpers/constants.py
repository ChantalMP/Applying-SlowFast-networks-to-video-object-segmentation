from pathlib import Path

model_name = 'model_maskrcnn_slowfast_residual_till_convergence'
models_dir_path = Path('models')
models_dir_path.mkdir(parents=True, exist_ok=True)
best_model_path = models_dir_path / (f'{model_name}_best.pth')
model_path = models_dir_path / (f'{model_name}.pth')
data_output_path = Path('data/output')
data_output_path.mkdir(parents=True, exist_ok=True)
eval_output_path = (data_output_path / 'eval_output')
eval_output_path.mkdir(parents=True, exist_ok=True)
pred_output_path = (data_output_path / 'pred_output')
pred_output_path.mkdir(parents=True, exist_ok=True)
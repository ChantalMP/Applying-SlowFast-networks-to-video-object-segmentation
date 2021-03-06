import torch

from helpers.constants import best_model_path, slow_pathway_size, fast_pathway_size
from helpers.evaluation import evaluate
from helpers.model import SegmentationModel


def predict_and_visualize():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: SegmentationModel = SegmentationModel(device=device, slow_pathway_size=slow_pathway_size,
                                                 fast_pathway_size=fast_pathway_size)
    model.to(device)
    model.load_state_dict(torch.load(best_model_path))
    evaluate(model, save_all_imgs=True)


if __name__ == '__main__':
    predict_and_visualize()

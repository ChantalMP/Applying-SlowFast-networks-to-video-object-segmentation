from helpers.model_old import SegmentationModel  # TODO change back to model
import torch
from helpers.evaluation import evaluate
from helpers.constants import model_path


def predict_and_visualize():
    slow_pathway_size = 4
    fast_pathway_size = 4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: SegmentationModel = SegmentationModel(device=device, slow_pathway_size=slow_pathway_size,
                                                 fast_pathway_size=fast_pathway_size)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    evaluate(model, save_all_imgs=True)

if __name__ == '__main__':
    predict_and_visualize()

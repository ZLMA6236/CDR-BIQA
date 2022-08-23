import torch
import numpy as np
import data_loader
import argparse
from PIL import Image
from models.swin_transformer import SwinTransformer
from scipy import stats


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


folder_path = {
    'livec': 'F:\DataSets\ChallengeDB_release\\',
    'bid': 'F:\DataSets\BID\ImageDatabases\\',
    'koniq-10k': 'F:\DataSets\koniq\\',
    'spaq': 'G:\DATASETS\SPAQ zip\SPAQ\\',
    'flive': r'E:\flive\\',
    'live': 'F:\DataSets\databaserelease\databaserelease2\\',
    'csiq': r'F:\DataSets\CSIQ\\',
}
img_num = {
    'livec': list(range(0, 1162)),
    'bid': list(range(0, 586)),
    'koniq-10k': list(range(0, 10073)),
    'spaq': list(range(0, 11125)),
    'flive': list(range(0, 39807)),
    'live': list(range(0, 982)),
    'csiq': list(range(0, 866)),
}


def build_loader_single(config):
    sel_num = img_num[config.dataset]
    path = folder_path[config.dataset]
    test_loader = data_loader.DataLoader(config.dataset, path, sel_num, config.patch_size, config.patch_num,
                                         istrain=False)
    data_loader_all = test_loader.get_data()

    return data_loader_all


@torch.no_grad()
def main(config):
    data_loader_all = build_loader_single(config)
    model = SwinTransformer().cuda()
    load_net = torch.load(config.pretrained_models)
    model.load_state_dict(load_net['model'], strict=False)
    model.eval()

    pred_scores = []
    gt_scores = []
    print('Testing %s dataset' % config.dataset)
    print('--------------------------------------------')
    for idx, (images, target) in enumerate(data_loader_all):
        images = images.clone().detach().cuda()
        target = target.clone().detach().cuda()
        output = model(images)
        pred_scores = pred_scores + output.cpu().tolist()
        gt_scores = gt_scores + target.cpu().tolist()
        if idx % 100 == 0:
            print('Testing: %d/ %d' % (idx, len(data_loader_all)))

    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, config.patch_num)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, config.patch_num)), axis=1)
    test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

    print('Testing %s dataset result: SRCC %.4f\tPLCC %.4f' % (config.dataset, test_srcc, test_plcc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='spaq',
                        help='Support datasets: livec|koniq-10k|bid|spaq|flive|live|csiq')
    parser.add_argument('--patch_num', dest='patch_num', type=int, default=10,
                        help='Number of sample patches from testing image')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for testing image patches')
    parser.add_argument('--pretrained_models', dest='pretrained_models', type=str,
                        default=r'./pretrained_models/CDR-BIQA.pth',
                        help='pretrained_models')

    config = parser.parse_args()
    main(config)

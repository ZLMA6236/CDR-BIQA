
import torch
import torchvision
from PIL import Image
import numpy as np
from models.swin_transformer import SwinTransformer


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


im_path = r'./data/test_image/4.jpg'
load_path = r'./pretrained_models/CDR-BIQA.pth'

model = SwinTransformer().cuda()
load_net = torch.load(load_path)
model.load_state_dict(load_net['model'], strict=False)
model.eval()


transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(384),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])

pred_scores = []
for i in range(10):
    img = pil_loader(im_path)
    img = transforms(img)
    img = img.unsqueeze(0).clone().detach().cuda()
    pred = model(img)
    pred_scores.append(float(pred.item()))
score = np.mean(pred_scores)

print('Predicted quality score: %.4f' % score)


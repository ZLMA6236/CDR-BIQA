import os
import time
import datetime
import numpy as np
from scipy import stats
import torch
import torch.backends.cudnn as cudnn
from timm.utils import AverageMeter
from config_swintiqa import get_config
from models import build_model
from data import build_loader_single
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, load_pretrained

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def parse_option():
    config = get_config()
    return config

def main(config):

    data_loader_train, data_loader_val = build_loader_single(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    # logger.info(str(model))
    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = torch.nn.L1Loss().cuda()

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        test_srcc, test_plcc, loss = validate(config, data_loader_val, model)
        logger.info(f"test_srcc test images: {test_srcc:.4f},test_plcc test images: {test_plcc:.4f}")
        logger.info("Continue training")

    start_time = time.time()
    max_srcc = 0.0
    corr_plcc = 0.0
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler)
        save_checkpoint(config, epoch, model, optimizer, lr_scheduler, logger)
        test_srcc, test_plcc, loss = validate(config, data_loader_val, model)
        logger.info(f"test_srcc test images: {test_srcc:.4f},test_plcc test images: {test_plcc:.4f}")

        if max_srcc < test_srcc:
            max_srcc = test_srcc
            corr_plcc = test_plcc

        logger.info(f'Max test_srcc: {max_srcc:.4f},Max test_plcc: {corr_plcc:.4f}')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    pred_scores = []
    gt_scores = []
    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.clone().detach().cuda()
        targets = targets.clone().detach().cuda()

        outputs = model(samples)
        loss = criterion(outputs.squeeze(), targets.float().detach())
        optimizer.zero_grad()
        if config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(amp.master_params(optimizer))
        else:
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        pred_scores = pred_scores + outputs.cpu().tolist()
        gt_scores = gt_scores + targets.cpu().tolist()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.TRAIN_PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.8f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})')
    loss_meter.reset()
    epoch_time = time.time() - start
    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)
    train_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    logger.info(f"train_srcc: {train_srcc:.4f},train_plcc: {train_plcc:.4f}")
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.L1Loss().cuda()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    pred_scores = []
    gt_scores = []
    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.clone().detach().cuda()
        target = target.clone().detach().cuda()
        output = model(images)
        loss = criterion(output.squeeze(), target.float().detach())
        pred_scores = pred_scores + output.cpu().tolist()
        gt_scores = gt_scores + target.cpu().tolist()
        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.TEST_PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')
    loss_meter.reset()
    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 10)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 10)), axis=1)
    test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    return test_srcc, test_plcc, loss_meter.avg


if __name__ == '__main__':
    config = parse_option()
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    seed = config.SEED + 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # print config
    logger.info(config.dump())
    main(config)

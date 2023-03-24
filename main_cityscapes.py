# import logging
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from data import CityScapes
from models import SegNetMtan

from utils import ConfMatrix, delta_fn_cityscapes, depth_error
from common import (
    common_parser,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from weight_methods import WeightMethods

from utils import create_log_dir
from pathlib import Path
import time
from adatask import Adam_with_AdaTask
from torch.optim import Adam
set_logger()

def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss


def main(path, lr, bs, device):
    # ----
    # Nets
    # ---

    model = SegNetMtan()
    model = model.to(device)

    # weight method
    weight_method = WeightMethods(args.method, n_tasks=2, device=device)

    # optimizer
    if args.optimizer == 'adam_with_adatask':
        optimizer = Adam_with_AdaTask([dict(params=model.parameters(), lr=lr)], n_tasks=2, args=args, device=device)
    elif args.optimizer == 'adam':
        optimizer = Adam([dict(params=model.parameters(), lr=lr)])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    train_set = CityScapes(root=path.as_posix(), train=True, augmentation=args.apply_augmentation)
    test_set = CityScapes(root=path.as_posix(), train=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False)

    # dataset and dataloaders
    log_str = ("Applying data augmentation." if args.apply_augmentation else "Standard training strategy without data augmentation.")
    logger.info(log_str)

    epochs = args.n_epochs
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 12], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(model.segnet.class_nb)
    logger.info('---train begin---')
    for epoch in range(epochs):
        cost = np.zeros(12, dtype=np.float32)

        for j, batch in enumerate(train_loader):
            custom_step += 1

            model.train()
            optimizer.zero_grad()

            train_data, train_label, train_depth = batch
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth = train_depth.to(device)

            train_pred, features = model(train_data, return_representation=True)

            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                )
            )

            if args.optimizer == 'adam_with_adatask':
                optimizer.backward_and_step(
                    losses=losses,
                    shared_parameters=list(model.shared_parameters()),
                    task_specific_parameters=list(model.task_specific_parameters()),
                    last_shared_parameters=list(model.last_shared_parameters()),
                )

            elif args.optimizer == 'adam':
                weight_method.backward(
                    losses=losses,
                    shared_parameters=list(model.shared_parameters()),
                    task_specific_parameters=list(model.task_specific_parameters()),
                    last_shared_parameters=list(model.last_shared_parameters()),
                    representation=features,
                )

                optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            avg_cost[epoch, :6] += cost[:6] / train_batch

            if j % 100 == 0:
                print(
                    f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.16f}, "
                    f"depth loss: {losses[1].item():.16f}, "
                )

        # scheduler
        scheduler.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # todo: move evaluate to function?
        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth = test_depth.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_pred[1], test_depth)
                avg_cost[epoch, 6:] += cost[6:] / test_batch

            # compute mIoU and acc
            avg_cost[epoch, 7:9] = conf_mat.get_metrics()

            # print results
            logger.info(f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR (test)")
            logger.info(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} "
                f"|| TEST: {avg_cost[epoch, 6]:.4f} {avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} "
                f"| {avg_cost[epoch, 9]:.4f} {avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f} "
                )


if __name__ == "__main__":
    parser = ArgumentParser("cityscapes", parents=[common_parser])
    parser.set_defaults(
        data_path='./dataset/cityscapes',
        log_path='./log/',
        batch_size=8,
        n_task=2,
        lr=1e-4,
        n_epochs=200,
    )
    parser.add_argument(
        "--method",
        type=str,
        default="equalweight",
        choices=["equalweight"],
        help="method type",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam_with_adatask",
        choices=["adam", "adam_with_adatask"],
        help="optimizer type",
    )
    parser.add_argument(
        "--apply-augmentation", type=str2bool, default=True, help="data augmentations"
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    logger = create_log_dir(str(Path(args.log_path)) + '/' + 'model_mtan' + '_optimizer_' + args.optimizer+'_' + time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
    logger.info(str(args))

    device = get_device(gpus=args.gpu)
    main(path=args.data_path, lr=args.lr, bs=args.batch_size, device=device)


import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import importlib, time
import traceback
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from utils.config import TrainConfig
from utils.logging import Logger, class_name_to_file_name, mkdir, cp_files
from utils.torch import get_device, save_state, load_state
from utils.stats import StatTracker

NUM_WORKERS = 2

def parse_args(argv):
    # create config and parse args
    config = TrainConfig(argv)
    known_args, unknown_args = config.parse()
    print('Unrecognized args: ' + str(unknown_args))
    return known_args

def train(args_obj, config_file):

    # set up output
    args = args_obj.base
    mkdir(args.out)

    # create logging system
    train_log_path = os.path.join(args.out, 'train.log')
    Logger.init(train_log_path)

    # save arguments used
    Logger.log('Base args: ' + str(args))
    Logger.log('Model args: ' + str(args_obj.model))
    Logger.log('Dataset args: ' + str(args_obj.dataset))
    Logger.log('Loss args: ' + str(args_obj.loss))

    # save training script/model/dataset used
    train_scripts_path = os.path.join(args.out, 'train_scripts')
    mkdir(train_scripts_path)
    pkg_root = os.path.join(cur_file_path, '..')
    dataset_file = class_name_to_file_name(args.dataset)
    dataset_file_path = os.path.join(pkg_root, 'datasets/' + dataset_file + '.py')
    model_file = class_name_to_file_name(args.model)
    loss_file = class_name_to_file_name(args.loss)
    model_file_path = os.path.join(pkg_root, 'models/' + model_file + '.py')
    train_file_path = os.path.join(pkg_root, 'train/train_humor.py')
    cp_files(train_scripts_path, [train_file_path, model_file_path, dataset_file_path, config_file])

    # load model class and instantiate
    model_class = importlib.import_module('models.' + model_file)
    Model = getattr(model_class, args.model)
    model = Model(**args_obj.model_dict,
                    model_smpl_batch_size=args.batch_size) # assumes model is HumorModel

    # load loss class and instantiate
    loss_class = importlib.import_module('losses.' + loss_file)
    Loss = getattr(loss_class, args.loss)
    loss_func = Loss(**args_obj.loss_dict,
                     smpl_batch_size=args.batch_size*args_obj.dataset.sample_num_frames) # assumes loss is HumorLoss

    device = get_device(args.gpu)
    model.to(device)
    loss_func.to(device)

    print(model)

    # count params in model
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    Logger.log('Num model params: ' + str(params))

    # freeze params in loss
    for param in loss_func.parameters():
        param.requires_grad = False

    # optimizer
    betas = (args.beta1, args.beta2)
    if args.use_adam:
        optimizer = optim.Adam(model.parameters(),
                                lr=args.lr,
                                betas=betas,
                                eps=args.eps,
                                weight_decay=args.decay)
    else:
        optimizer = optim.Adamax(model.parameters(),
                                lr=args.lr,
                                betas=betas,
                                eps=args.eps,
                                weight_decay=args.decay)

    # load in pretrained weights/optimizer state if given
    start_epoch = 0
    min_val_loss = min_train_loss = float('inf')
    if args.ckpt is not None:
        load_optim = optimizer if args.load_optim else None
        start_epoch, min_val_loss, min_train_loss = load_state(args.ckpt, model, optimizer=load_optim, map_location=device, ignore_keys=model.ignore_keys)
        start_epoch += 1
        Logger.log('Resuming from saved checkpoint at epoch idx %d with min val loss %.6f...' % (start_epoch, min_val_loss))
        if not args.load_optim:
            Logger.log('Not loading optimizer state as desired...')
            Logger.log('WARNING: Also resetting min_val_loss and epoch count!')
            min_val_loss = float('inf')
            start_epoch = 0

    # initialize LR scheduler
    scheduler = MultiStepLR(optimizer, milestones=args.sched_milestones, gamma=args.sched_decay)

    # intialize schedule sampling if desired
    use_sched_samp = False
    if args.sched_samp_start is not None and args.sched_samp_end is not None:
        if args.sched_samp_start >= 0 and args.sched_samp_end >= args.sched_samp_start:
            Logger.log('Using scheduled sampling starting at epoch %d and ending at epoch %d!' % (args.sched_samp_start, args.sched_samp_end))
            use_sched_samp = True
        else:
            Logger.log('Could not use scheduled sampling with given start and end!')

    # load dataset class and instantiate training and validation set
    Dataset = getattr(importlib.import_module('datasets.' + dataset_file), args.dataset)
    train_dataset = Dataset(split='train', **args_obj.dataset_dict)
    val_dataset = Dataset(split='val', **args_obj.dataset_dict)
    # create loaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True,
                              worker_init_fn=lambda _: np.random.seed()) # get around pytorch RNG seed bug
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size,
                            shuffle=False, 
                            num_workers=NUM_WORKERS,
                            pin_memory=True,
                            worker_init_fn=lambda _: np.random.seed())

    # stats tracker
    tensorboard_path = os.path.join(args.out, 'train_tensorboard')
    mkdir(tensorboard_path)
    stat_tracker = StatTracker(tensorboard_path)

    # checkpoints saving
    ckpts_path = os.path.join(args.out, 'checkpoints')
    mkdir(ckpts_path)

    if use_sched_samp:
        train_dataset.return_global = True
        val_dataset.return_global = True

    # main training loop
    train_start_t = time.time()
    for epoch in range(start_epoch, args.epochs):

        model.train()

        # train
        stat_tracker.reset()
        batch_start_t = None
        reset_loss_track = train_dataset.pre_batch(epoch=epoch)
        # see which phase we're in 
        sched_samp_gt_p = 1.0 # supervised
        if use_sched_samp:
            if epoch >= args.sched_samp_start and epoch < args.sched_samp_end:
                frac = (epoch - args.sched_samp_start) / (args.sched_samp_end - args.sched_samp_start)
                sched_samp_gt_p = 1.0*(1.0 - frac)
            elif epoch >= args.sched_samp_end:
                # autoregressive
                sched_samp_gt_p = 0.0
            Logger.log('Scheduled sampling current use_gt_p = %f' % (sched_samp_gt_p))

            if epoch == args.sched_samp_end:
                # the loss will naturally go up when using own rollouts
                reset_loss_track = True

            if args_obj.loss_dict['kl_loss_cycle_len'] > 0:
                # if we're cycling, only want to save results when using full ELBO
                if (epoch % args_obj.loss_dict['kl_loss_cycle_len']) > (args_obj.loss_dict['kl_loss_cycle_len'] // 2):
                    # have reached second half of a cycle
                    reset_loss_track = True

        if reset_loss_track:
            Logger.log('Resetting min_val_loss and min_train_loss')
            min_val_loss = min_train_loss = float('inf')

        for i, data in enumerate(train_loader):
            batch_start_t = time.time()

            try:
                # zero the gradients
                optimizer.zero_grad()
                # run model
                loss, stats_dict = model_class.step(model, loss_func, data, train_dataset, device, epoch, mode='train', use_gt_p=sched_samp_gt_p)
                if torch.isnan(loss).item():
                    Logger.log('WARNING: NaN loss. Skipping to next data...')
                    torch.cuda.empty_cache()
                    continue
                # backprop and step
                loss.backward()
                # check gradients
                parameters = [p for p in model.parameters() if p.grad is not None]
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
                if torch.isnan(total_norm):
                    Logger.log('WARNING: NaN gradients. Skipping to next data...')
                    torch.cuda.empty_cache()
                    continue
                optimizer.step()
            except (RuntimeError, AssertionError) as e:
                if epoch > 0:
                    # to catch bad dynamics, but keep training
                    Logger.log('WARNING: caught an exception during forward or backward pass. Skipping to next data...')
                    Logger.log(e)
                    traceback.print_exc()
                    reset_loss_track = train_dataset.pre_batch(epoch=epoch)
                    if reset_loss_track:
                        Logger.log('Resetting min_val_loss and min_train_loss')
                        min_val_loss = min_train_loss = float('inf')
                    continue
                else:
                    raise e

            # collect stats
            batch_elapsed_t = time.time() - batch_start_t
            total_elapsed_t = time.time() - train_start_t
            stats_dict['loss'] = loss
            for param_group in optimizer.param_groups:
                stats_dict['lr'] = torch.Tensor([param_group['lr']])[0]
            stats_dict['time_per_batch'] = torch.Tensor([batch_elapsed_t])[0]

            last_batch = (i==(len(train_loader)-1))
            stat_tracker.update(stats_dict, tag='train', save_tf=last_batch)
            if i % args.print_every == 0:
                stat_tracker.print(i, len(train_loader),
                                epoch, args.epochs,
                                total_elapsed_time=total_elapsed_t,
                                tag='train')

            reset_loss_track = train_dataset.pre_batch(epoch=epoch)
            if reset_loss_track:
                Logger.log('Resetting min_val_loss and min_train_loss')
                min_val_loss = min_train_loss = float('inf')

        # save if desired
        if epoch % args.save_every == 0:
            Logger.log('Saving checkpoint...')
            save_file = os.path.join(ckpts_path, 'epoch_%08d_model.pth' % (epoch))
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_val_loss, min_train_loss=min_train_loss, ignore_keys=model.ignore_keys)

        # check if it's the best train model so far
        mean_train_loss = stat_tracker.meter_dict['train/loss'].avg
        if mean_train_loss < min_train_loss:
            min_train_loss = mean_train_loss
            Logger.log('Best train loss so far! Saving checkpoint...')
            save_file = os.path.join(ckpts_path, 'best_train_model.pth')
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_val_loss, min_train_loss=min_train_loss, ignore_keys=model.ignore_keys)

        # validate
        if epoch % args.val_every == 0:
            with torch.no_grad():
                # run on validation data
                model.eval()

                stat_tracker.reset()
                for i, data in enumerate(val_loader):
                    # print(i)
                    batch_start_t = time.time()
                    # run model
                    loss, stats_dict = model_class.step(model, loss_func, data, val_dataset, device, epoch, mode='test', use_gt_p=sched_samp_gt_p)

                    if torch.isnan(loss):
                        Logger.log('WARNING: NaN loss on VALIDATION. Skipping to next data...')
                        continue

                    # collect stats
                    batch_elapsed_t = time.time() - batch_start_t
                    total_elapsed_t = time.time() - train_start_t
                    stats_dict['loss'] = loss
                    stats_dict['time_per_batch'] = torch.Tensor([batch_elapsed_t])[0]

                    stat_tracker.update(stats_dict, tag='val', save_tf=(i==(len(val_loader)-1)), increment_step=False)

                    if i % args.print_every == 0:
                        stat_tracker.print(i, len(val_loader),
                                        epoch, args.epochs,
                                        total_elapsed_time=total_elapsed_t,
                                        tag='val')

                # check if it's the best model so far
                mean_val_loss = stat_tracker.meter_dict['val/loss'].avg
                if mean_val_loss < min_val_loss:
                    min_val_loss = mean_val_loss
                    Logger.log('Best val loss so far! Saving checkpoint...')
                    save_file = os.path.join(ckpts_path, 'best_model.pth')
                    save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_val_loss, min_train_loss=min_train_loss, ignore_keys=model.ignore_keys)

        scheduler.step()

        torch.cuda.empty_cache()

    Logger.log('Finished!')

def main(args, config_file):
    train(args, config_file)

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    config_file = sys.argv[1:][0][1:]
    main(args, config_file)
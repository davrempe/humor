
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime, timedelta

from utils.logging import Logger

class AverageMeter(object):
    """Computes and stores the average and current scalar value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class VectorMeter(object):
    """
    Stores all values that are given as vectors
    so can compute things like median/std.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = []

    def update(self, val):
        self.val += val.tolist()

    def mean(self):
        return np.mean(np.array(self.val))
    
    def std(self):
        return np.std(np.array(self.val))

    def median(self):
        return np.median(np.array(self.val))

def dhms(td):
    d, h, m = td.days, td.seconds//3600, (td.seconds//60)%60
    s = td.seconds - ( (h*3600) + (m*60) ) # td.seconds are the seconds remaining after days have been removed
    return d, h, m, s

def getTimeDur(seconds):
    Duration = timedelta(seconds=seconds)
    OutStr = ''
    d, h, m, s = dhms(Duration)
    if d > 0:
        OutStr = OutStr + str(d)+ ' d '
    if h > 0:
        OutStr = OutStr + str(h) + ' h '
    if m > 0:
        OutStr = OutStr + str(m) + ' m '
    OutStr = OutStr + str(s) + ' s'

    return OutStr

class StatTracker(object):
    '''
    Keeps track of stats of desired stats throughout training/testing.

    This includes a running mean and tensorboard visualization output.
    '''

    def __init__(self, out_dir):
        self.writer = SummaryWriter(out_dir)
        self.step = 0
        self.meter_dict = dict()
        self.vector_meter_dict = dict()

    def reset(self):
        # keep global track for tensorboard but only per-epoch for meter_dict
        self.meter_dict = dict()

    def update(self, stats_dict, tag='train', save_tf=True, n=1, increment_step=True):
        all_tag = 'run'
        # find stats of each type
        scalar_dict = dict()
        vector_dict = dict()
        image_dict = dict()
        pcl_dict = dict()
        for k in stats_dict.keys():
            if torch.is_tensor(stats_dict[k]):
                num_dims = len(stats_dict[k].size())
            else:
                stats_dict[k] = torch.Tensor([stats_dict[k]])[0]
                num_dims = 0
            # print('%s : %d' % (k, num_dims))
            if num_dims == 0:
                # scalar
                scalar_dict[tag + '/' + k] = stats_dict[k].cpu().item()
            elif num_dims == 1:
                # vector
                vector_dict[tag + '/' + k] = stats_dict[k].cpu().data.numpy()
                scalar_dict[tag + '/' + k] = vector_dict[tag + '/' + k].mean()
            elif num_dims == 2:
                # point cloud
                pcl_dict[tag + '/' + k] = stats_dict[k].cpu().unsqueeze(0)
            elif num_dims == 3:
                # image
                image_dict[tag + '/' + k] = stats_dict[k].cpu()

        # update average meter dicts
        for k, v in scalar_dict.items():
            if not k in self.meter_dict:
                self.meter_dict[k] = AverageMeter()
            self.meter_dict[k].update(v, n=n)
            # update scalar dict for tf save
            scalar_dict[k] = self.meter_dict[k].avg

         # update vector meter dicts
        for k, v in vector_dict.items():
            if not k in self.vector_meter_dict:
                self.vector_meter_dict[k] = VectorMeter()
            self.vector_meter_dict[k].update(v)

        # write to tensorboard
        if save_tf:
            self.writer.add_scalars(all_tag, scalar_dict, self.step)
            for k, v in image_dict.items():
                self.writer.add_image(all_tag + '/' + k, v, self.step)
            for k, v in pcl_dict.items():
                colors = 255*((v / torch.max(v)) + 0.5)
                points_config = {
                    'cls': 'PointsMaterial',
                    'size': 0.05
                    }
                self.writer.add_mesh(all_tag + '/' + k, v, colors.to(torch.int), global_step=self.step, config_dict={'material' : points_config})

        if increment_step:
            self.step += 1
        

    def print(self, cur_batch_idx, num_batches, cur_epoch_idx, num_epochs, total_elapsed_time=None, tag='train'):
        # print the progress bar with estimated time
        done = int(50 * (cur_batch_idx+1) / num_batches)
        progress_str = '[{}>{}] {} epoch - {}/{} | batch - {}/{}'.format('=' * done, '-' * (50 - done), tag,
                                                                        cur_epoch_idx+1, num_epochs, 
                                                                        cur_batch_idx+1, num_batches)
        Logger.log(progress_str)

        # timing stats if available
        time_per_batch_str = tag + '/' + 'time_per_batch'
        if time_per_batch_str in self.meter_dict and total_elapsed_time is not None:
            mean_per_batch = self.meter_dict[time_per_batch_str].avg
            elapsed = total_elapsed_time
            elapsed_str = getTimeDur(elapsed)
            cur_frac = (num_batches*cur_epoch_idx + cur_batch_idx) / (num_batches*num_epochs)
            ETA = (elapsed / (cur_frac + 1e-6)) - elapsed
            ETA_str = getTimeDur(ETA)
            time_str = '%.3f s per batch | %s elapsed | %s ETA' % (mean_per_batch, elapsed_str, ETA_str)
            Logger.log(time_str)

        # recorded stats
        for k, v in self.meter_dict.items():
            stat_str = '%s : %.5f' % (k, v.avg)
            # see if there's an associated vector value
            if k in self.vector_meter_dict:
                vec_meter = self.vector_meter_dict[k]
                stat_str += ' mean, %.5f std, %.5f med' % (vec_meter.std(), vec_meter.median())
            Logger.log(stat_str)

    

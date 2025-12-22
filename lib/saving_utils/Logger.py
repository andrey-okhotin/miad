import os
import time
from collections import defaultdict

import torch

from saving_utils.get_repo_root import get_repo_root




class Logger:
    
    def __init__(self, name, process, new_logs=True):
        self.process = process
        self.path_progress_logs = os.path.join(
            get_repo_root(), 'logs', 'progress_logs', f'{name}.txt')
        self.path_logs = os.path.join(
            get_repo_root(), 'logs', 'logs', f'{name}.pt')
        
        # reset previous
        if self.process.is_root_process:
            if new_logs:
                if os.path.exists(self.path_logs):
                    os.remove(self.path_logs)
                self.reset()
        else:
            time.sleep(5)
        
        # load existed logs
        self.logs = defaultdict(list)
        if os.path.exists(self.path_logs) and not new_logs:
            self.logs = torch.load(self.path_logs, map_location='cpu')
        pass
    
    
    
    
    def add(self, key, log, stack_after_epoch=False):
        if not self.process.is_root_process:
            return None
        if key in self.logs:
            if stack_after_epoch:
                if isinstance(self.logs[key][-1], list):
                    self.logs[key][-1].append(log)
                else:
                    self.logs[key].append([log])
            else:
                self.logs[key].append(log)
        else:
            if stack_after_epoch:
                self.logs[key].append([log])
            else:
                self.logs[key].append(log)
        pass
    
    
    
    
    def stack_logs(self):
        if not self.process.is_root_process:
            return None
        for key, logs in self.logs.items():
            if isinstance(self.logs[key][-1], list):
                self.logs[key][-1] = torch.tensor(
                    list(map(
                        lambda t: t.tolist() if isinstance(t, torch.Tensor) else t,
                        self.logs[key][-1]
                    )),
                    dtype=torch.float32
                ).reshape(len(self.logs[key][-1]), -1).mean(dim=0)
        pass
        
        
        
        
    def save(self):
        if not self.process.is_root_process:
            return None
        torch.save(self.logs, self.path_logs)
        pass
    
    
    
    
    def reset(self):
        with open(self.path_progress_logs, 'w') as f:
            f.write(
                '\n'+
                '-----------------------------\n'+
                '|      PROGRESS LOGGER      |\n'+
                '-----------------------------\n'+
                '\n'
            )
        pass
    
    
    
    
    def fprint(self, *args, **kwargs):
        with open(self.path_progress_logs, 'a') as f:
            for message in args:
                message = str(message) + " "
                f.write(message)
            if 'end' in kwargs.keys():
                f.write(kwargs['end'])
            else:
                f.write('\n')
        pass




    def root_fprint(self, *args, **kwargs):
        if self.process.is_root_process:
            self.fprint(*args, **kwargs)

    

    
    def train_logs_pruning(self, start_epoch):
        if not self.process.is_root_process:
            return None
        iters_per_epoch = 0
        try:
            with open(self.path_progress_logs, 'r') as f:
                iters_per_epoch = int(f.read().split('\n')[10].split('/')[-1])
        except:
            pass
        eval_freq = self.logs['config'][0].eval_freq
        failed_epoch = self.logs['epoch'][-1] + 1
        for k, v in self.logs.items():
            # logs from every iteration
            if (failed_epoch + 1) * iters_per_epoch >= len(v) >= (start_epoch - 1) * iters_per_epoch:
                self.logs[k] = v[:(start_epoch - 1) * iters_per_epoch]
            # logs from every epoch
            elif (failed_epoch + 1) >= len(v) >= (start_epoch - 1):
                self.logs[k] = v[:(start_epoch - 1)]
            # logs from every eval
            elif (failed_epoch + 1) // eval_freq >= len(v) >= start_epoch // eval_freq:
                self.logs[k] = v[:(start_epoch // eval_freq)]
        pass

        
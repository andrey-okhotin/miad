# exec "run.py" only from "miad" folder

import os
import sys
import argparse
import traceback
import time
import warnings
import importlib
from datetime import timedelta

import random
import numpy
import yaml
from ml_collections import ConfigDict

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

sys.path.append(os.path.join(os.getcwd(), 'lib'))

from saving_utils.Logger import Logger
from saving_utils.config_yaml import load_config_yaml




def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    pass




def pipelines_runner(rank, args):
    # load config
    try:
        with open(os.path.join('saved_configs', args.config), 'r') as f:
            pipeline_config = ConfigDict(yaml.safe_load(f))
        pipeline_config.args = args
    except:
        raise FileNotFoundError(
            f"There is no config {os.path.join('saved_configs', args.config)} in folder 'saved_configs'"
        )

    # multiprocessing
    process = ConfigDict()
    process.rank = rank
    process.world_size = args.world_size
    process.is_root_process = (process.rank == 0)
    process.distributed = args.world_size > 1
    process.seed = pipeline_config.seed + process.rank
    process.gpu = args.gpu[process.rank] if len(args.gpu) else 'cpu'
    if process.distributed:
        dist.init_process_group(
            backend='nccl',
            world_size=process.world_size,
            rank=process.rank,
            timeout=timedelta(hours=10)
        )
    if args.ignore_warnings:
        warnings.filterwarnings('ignore')

    # init logger
    if 'continue_training' in pipeline_config:
        logger = Logger(pipeline_config.name, process, new_logs=False)
        logger.fprint(f'\nCONTINUE LOGS')
    else:
        logger = Logger(pipeline_config.name, process, new_logs=True)
    logger.add('config', pipeline_config)
    logger.save()

    # load pipeline runner
    pipeline_module = importlib.import_module(f"pipelines.{pipeline_config.pipeline}")
    
    # for neat output
    logger.fprint(f'START >> proc: {process.rank} device: {process.gpu}')
    if process.distributed:
        dist.barrier()
    logger.root_fprint("")
    if process.distributed:
        dist.barrier()

    # field that will be visible from all files
    os.environ['MODIFICATIONS_FIELD'] = pipeline_config.modifications
    # [!] noinspection PyBroadException
    try:
        fix_seed(process.seed)
        # RUN MAIN SCRIPT
        pipeline_module.run(process, pipeline_config, logger)
    except Exception as e:
        time.sleep(10 * process.rank)
        logger.fprint(traceback.format_exc())
        time.sleep(10 * (process.world_size - process.rank))
        if process.distributed:
            dist.destroy_process_group()
    del os.environ['MODIFICATIONS_FIELD']

    # for neat output
    if process.distributed:
        dist.barrier()
    logger.root_fprint("")
    if process.distributed:
        dist.barrier()
    logger.fprint(f'END >> proc: {process.rank} device: {process.gpu}')
    if process.distributed:
        dist.barrier()
    logger.root_fprint("\n")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', default=None, type=str)
    parser.add_argument('-gpu', '--gpu', default=None, type=str)
    parser.add_argument('-ignore_warnings', '--ignore_warnings', default=0, type=int)
    args = parser.parse_args()

    if args.gpu is not None:
        args.gpu = tuple(map(int, args.gpu.split('_')))
    else:
        args.gpu = []
    args.world_size = max(1, len(args.gpu))

    # for disabling tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # mp.freeze_support()

    print('Run with args:', args, flush=True)
    if args.world_size > 1:
        # spawn processes with defined pipeline
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        mp.set_start_method('spawn')
        # check possible ports
        for port in range(8900, 8999):
            os.environ['MASTER_PORT'] = str(port)
            try:
                mp.spawn(pipelines_runner, nprocs=args.world_size, args=(args,))
                break
            except RuntimeError:
                continue
    else:
        pipelines_runner(0, args)




if __name__ == '__main__':
    main()

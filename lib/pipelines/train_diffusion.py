import gc

import torch
import torch.distributed as dist

from visualization_utils.ProgressPrinters import trainer_printer
from saving_utils.get_repo_root import get_repo_root

from data.init_data import init_data
from diffusion.init_diffusion import init_diffusion
from models.init_model import init_model
from models.to_distributed import to_distributed

from optimizers.Optimizers import Optimizer
from saving_utils.Logger import Logger
from saving_utils.Checkpoints import Checkpoints




def run(process, config, logger):

    # init main objects

    data = init_data(config.data, process)
    diffusion = init_diffusion(config.diffusion, logger)
    model = init_model(config.model)
    optimizer = Optimizer(model, config.optimization)
    model_checkpoints = Checkpoints(config.checkpoints, process)

    # continue training case
        
    if 'continue_training' in config:
        model_checkpoints.load_checkpoint(
            model, 
            optimizer,
            config.continue_training.start_epoch
        )
        logger.train_logs_pruning(config.continue_training.start_epoch)
        logger.fprint(f'process {process.rank} load checkpoint {config.continue_training.start_epoch}')
        start_epoch = config.continue_training.start_epoch
    else:
        start_epoch = 0
    
    # device operations
    
    torch.cuda.set_device(process.gpu)
    model.cuda(process.gpu)
    optimizer.cuda(process.gpu)
    diffusion.cuda(process.gpu)
    torch.cuda.empty_cache()
    
    # distributed dataloader

    num_batches_in_epoch = len(data.dataloader_train.dataset) // data.config.batch_size
    if process.distributed:
        model = to_distributed(model, process.gpu)
        data.to_distributed('train')
        dist.barrier()
        dataloader = data.distributed_dataloader_train
    else:
        dataloader = data.dataloader_train
    logger.fprint(f'process {process.rank} bs: {dataloader.batch_size}')
    
    # main loop
    
    for epoch in range(start_epoch, config.n_epochs+1):

        # train stage
            
        model.train()
        if process.distributed:
            data.distributed_sampler.set_epoch(epoch)

        for batch_index, torch_geometric_batch in enumerate(dataloader):
            trainer_printer(
                process.rank, 
                epoch, config.n_epochs, 
                batch_index, num_batches_in_epoch, 
                fprint=logger.fprint
            )
            batch = data.ab_init_gen_training_batch(torch_geometric_batch, device='cuda')
            diffusion.train_step(
                batch=batch,
                model=model,
                mode='train'
            )
            optimizer.optimizing_step(
                loss=batch['loss'],
                logger=logger
            )
            logger.add("loss:train", batch['loss'].item(), stack_after_epoch=True)
            del batch
        
        gc.collect()
        torch.cuda.empty_cache()

        if process.distributed:
            dist.barrier()

        # eval stage

        if process.is_root_process and ((epoch+1) % config.eval_freq == 0):
            logger.root_fprint(f'\n<> epoch: {epoch:6d}', flush=True)
            optimizer.switch_to_ema()
            model.eval()
            with torch.no_grad():
                
                # go through train, validation and test parts of dataset with fixed model params
                
                for part in data.config.parts:
                    mode = f"{part}_eval"
                    eval_dataloader = getattr(data, f"dataloader_{part}")
                    logger.fprint(f'<> eval stage: mode -> {mode}', flush=True)

                    for batch_index, torch_geometric_batch in enumerate(eval_dataloader):
                        batch = data.ab_init_gen_training_batch(torch_geometric_batch, device='cuda')
                        diffusion.train_step(
                            batch=batch,
                            model=model,
                            mode=mode
                        )
                        logger.add(f"loss:{mode}", batch['loss'].item(), stack_after_epoch=True)
                        del batch

                    gc.collect()
                    torch.cuda.empty_cache()

            optimizer.switch_from_ema()
            logger.add("epoch", epoch)
            logger.stack_logs()
            logger.save()
        
        logger.stack_logs()
        if (epoch+1) % 10 == 0:
            logger.save()
        model_checkpoints.save_checkpoint(
            model, 
            optimizer, 
            epoch
        )
        
        if process.distributed:    
            dist.barrier()
        
        # end of the epoch

    optimizer.switch_to_ema()
    
    model.eval()
    model.cpu()
    diffusion.cpu()
    pass
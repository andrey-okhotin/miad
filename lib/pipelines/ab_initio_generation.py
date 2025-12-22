import os
import shutil

import torch

from saving_utils.get_repo_root import get_repo_root
from visualization_utils.ProgressPrinters import steps_printer

from data.init_data import init_data
from diffusion.init_diffusion import init_diffusion
from models.init_model import init_model




def run(process, config, logger):
    
    # init main objects

    data = init_data(config.data, process)
    diffusion = init_diffusion(config.diffusion, logger)
    model = init_model(config.model)
    model.eval()

    # created folder to save results

    main_folder = os.path.join(get_repo_root(), 'saved_results', config.name)
    folder = os.path.join(main_folder, 'generated_crystals_cif')
    if process.distributed:
        if process.is_root_process:
            if os.path.exists(main_folder):
                shutil.rmtree(main_folder)
            os.mkdir(main_folder)
            os.mkdir(folder)
        torch.distributed.barrier()
    else:
        if os.path.exists(main_folder):
            shutil.rmtree(main_folder)
        os.mkdir(main_folder)
        os.mkdir(folder)
    
    # process and gpu coordination

    torch.cuda.set_device(process.gpu)
    model.cuda(process.gpu)
    diffusion.cuda(process.gpu)
    
    # coordinate number of samples

    total_batch_size = min(config.num_samples, data.config.batch_size)
    batch_size_in_process = total_batch_size // process.world_size
    num_samples_in_process = config.num_samples // process.world_size
    if process.is_root_process:
        num_samples_in_process += config.num_samples % process.world_size
    num_iterations = num_samples_in_process // batch_size_in_process
    if num_samples_in_process % batch_size_in_process > 0:
        num_iterations += 1
    remaining_samples_in_process = num_samples_in_process

    # sampling loop

    torch.cuda.empty_cache()
    with torch.no_grad():
        for i in range(num_iterations):
            steps_printer.print_freq = diffusion.num_steps // 10
            progress_printer = lambda t_value: steps_printer(
                i, num_iterations, 
                t_value, diffusion.time_distribution.num_steps,
                process.rank, fprint=logger.fprint
            )
            bs = min(batch_size_in_process, remaining_samples_in_process)
            batch = data.ab_init_gen_sampling_batch(bs, device='cuda')
            batch = diffusion.sampling_procedure(
                model=model,
                batch=batch,
                progress_printer=progress_printer
            )
            data.save_batch(
                batch,
                folder=folder,
                rank=process.rank
            )
            logger.save()
            remaining_samples_in_process -= batch_size_in_process

    if process.is_root_process:
        shutil.make_archive(folder, 'zip', folder)
    
    model.cpu()
    diffusion.cpu()
    logger.save()
    pass



import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ml_collections import ConfigDict

from saving_utils.Logger import Logger




def load_logger(path):
    tmp_process = ConfigDict({ 'is_root_process' : True })
    logger = Logger(path, tmp_process, new_logs=False)
    return logger




def plot_loss(
    logger, 
    log=False,
    lower_bound=0.9,
    upper_bound=1.5
):
    style_config = ConfigDict({
        'ft_label'  : 16,
        'ft_legend' : 14
    })
    smooth = lambda stat: gaussian_filter1d(
        torch.tensor(stat).numpy(),
        sigma=50.,
        mode='nearest',
        radius=50
    )
    num_evals = len(logger.logs['loss:train_eval'])
    eval_freq = logger.logs['config'][0].eval_freq
    eval_epochs = eval_freq * (1 + torch.arange(num_evals)) - 1
    fig = plt.figure(figsize=(8,5))
    plt.ylabel('LOSS', fontsize=style_config.ft_label)
    plt.xlabel('EPOCH', fontsize=style_config.ft_label)
    plt.plot(logger.logs['loss:train'], 
             color='blue', alpha=0.1)
    plt.plot(smooth(logger.logs['loss:train']), 
             color='blue', label='train')
    plt.plot(eval_epochs, logger.logs['loss:train_eval'], 
             label='train:eval', color='green')
    plt.plot(eval_epochs, logger.logs['loss:validation_eval'], 
             label='validation:eval', color='red')
    if f'loss:test_eval' in logger.logs:
            plt.plot(eval_epochs, logger.logs[f'loss:test_eval'], 
                     label='test:eval', color='violet')
    plt.ylim(lower_bound, upper_bound)
    plt.grid()
    plt.legend(loc=1, fontsize=style_config.ft_legend)
    plt.show()
    

    
    
def plot_separate_losses(
    logger, 
    log=False,
    lower_bound=0.9,
    upper_bound=1.5,
    xlim=None,
    type_loss=False
):
    style_config = ConfigDict({
        'ft_label'  : 16,
        'ft_legend' : 14
    })
    smooth = lambda stat: gaussian_filter1d(
        torch.tensor(stat).numpy(),
        sigma=50.,
        mode='nearest',
        radius=50
    )
    num_evals = len(logger.logs['loss:train_eval'])
    eval_freq = logger.logs['config'][0].eval_freq
    eval_epochs = eval_freq * (1 + torch.arange(num_evals)) - 1
    fig = plt.figure(figsize=(21,5))
    for i, part in enumerate(['', ':lattice', ':coord', ':type']):
        if not (f'loss{part}:train' in logger.logs):
            continue
        ax = fig.add_subplot(131+i + 10*type_loss)
        plt.ylabel(f'LOSS{part}', fontsize=style_config.ft_label)
        plt.xlabel('EPOCH', fontsize=style_config.ft_label)
        plt.plot(logger.logs[f'loss{part}:train'], 
                 color='blue', alpha=0.1)
        plt.plot(smooth(logger.logs[f'loss{part}:train']), 
                 color='blue', label='train')
        plt.plot(eval_epochs, logger.logs[f'loss{part}:train_eval'], 
                 label='train:eval', color='green')
        plt.plot(eval_epochs, logger.logs[f'loss{part}:validation_eval'], 
                 label='validation:eval', color='red')
        if f'loss{part}:test_eval' in logger.logs:
            plt.plot(eval_epochs, logger.logs[f'loss{part}:test_eval'], 
                     label='test:eval', color='violet')
        plt.ylim(lower_bound, upper_bound)
        if not (xlim is None):
            plt.xlim(*xlim)
        plt.grid()
        plt.legend(loc='best', fontsize=style_config.ft_legend)
        if log:
            plt.yscale('log')
    plt.show()

    
    

def plot_loss4time(
    logger,
    key,
    stages = [ 0., 1. ],
    lower_bound = 0.7,
    upper_bound = 2.5,
    log=False
):
    style_config = ConfigDict({
        'ft_label'  : 14,
        'ft_legend' : 10
    })
    smooth = lambda stat: gaussian_filter1d(
        stat.numpy(),
        sigma=50.,
        mode='nearest',
        radius=50
    )
    config = logger.logs['config'][0]
    bs = config.data.batch_size // 2 + 1
    reverse_metric_scale = 1000 / bs
    alpha = 0.1
    fig = plt.figure(figsize=(20,5))
    for i, mode in enumerate([ 'train', 'train_eval', 'validation_eval', 'test_eval' ]):
        stat = logger.logs[f'loss:{key}:{mode}']
        stage_coef = 1 if mode == 'train' else config.eval_freq
        epoch_stages = (torch.tensor(stages) * (len(stat)-1)).to(int)
        epoch_colors = [(i/(len(stages)-1), 0, 1-i/(len(stages)-1)) 
                        for i in range(len(stages)) ]

        ax = fig.add_subplot(141+i)
        plt.title(mode, fontsize=style_config.ft_label)
        plt.ylabel('LOSS', fontsize=style_config.ft_label)
        plt.xlabel('TIME', fontsize=style_config.ft_label)
        for stage, color in zip(epoch_stages, epoch_colors):
            plt.plot(reverse_metric_scale * stat[stage], 
                     alpha=alpha, color=color)
        for stage, color in zip(epoch_stages, epoch_colors):
            plt.plot(reverse_metric_scale * smooth(stat[stage]), 
                     color=color, label=f'epoch = {stage_coef * (stage + 1)}')    
        plt.ylim(lower_bound, upper_bound)
        if log:
            plt.yscale('log')
        plt.legend(loc='best', fontsize=style_config.ft_legend)
    plt.show()


    
    
def plot_norms(folder_name):
    fig = plt.figure(figsize=(15,5))

    # check grad norm
    while 1:
        try:
            logger = torch.load(f'../logs/logs/{folder_name}.pt')
            break
        except:
            pass
    ax = fig.add_subplot(131)
    plt.title('grad:l2-norm')
    plt.plot(logger['grad:l2-norm'])
    plt.ylim(0,1)
    ax = fig.add_subplot(132)
    plt.title('grad:max-norm')
    plt.plot(logger['grad:max-norm'])
    plt.ylim(0,1)

    # check weights norm
    def state_dict_norm(f):
        p = []
        for _, t in f.items():
            if isinstance(t, torch.Tensor):
                p.append(t.flatten())
        return torch.cat(p).norm()

    ax = fig.add_subplot(133)
    plt.title('weight:l2-norm')
    norms = []
    
    for checkpoint in os.listdir(f'../checkpoints/{folder_name}'):
        if checkpoint[-3:] == '.pt':
            state_dict = torch.load(
                f'../checkpoints/{folder_name}/{checkpoint}',
                map_location=torch.device('cpu')
            )
            norms.append(state_dict_norm(state_dict).item())
    plt.plot(norms)

    plt.show()
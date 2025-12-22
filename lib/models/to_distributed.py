from torch.nn.parallel import DistributedDataParallel




def to_distributed(model, gpu, process_group=None):
    """
    """
    # save default model params
    model_config = model.model_config
    
    model = DistributedDataParallel(
        model, 
        device_ids=[gpu], 
        process_group=process_group
    )
    
    # load default model params to DDP
    model.model_config = model_config

    return model
    
    
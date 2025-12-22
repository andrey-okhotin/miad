from data.crystal_data_storage import CrystalDataStorage




def init_data(config, process):
    switch = {
        'CrystalDataStorage' : CrystalDataStorage
    }
    if config['type'] in switch:
        return switch[config['type']](config, process)
    raise NotImplementedError
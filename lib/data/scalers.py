import torch




def init_scaler(scaler_name, data_sample):
    switch = {
        'StandardContiniousScaler' : StandardContiniousScaler
    }
    return switch[scaler_name](data_sample)




class StandardContiniousScaler:

    def __init__(self, data_sample):
        data_sample = torch.vstack(data_sample)
        self.mean = data_sample.mean(dim=0)
        self.std = data_sample.std(dim=0)
        self.eps = 1e-3

    def align_devices(self, data):
        self.mean, self.std = self.mean.to(data.device), self.std.to(data.device)
        return data

    def rescale(self, data):
        data = self.align_dtypes_and_devices(data)
        return (data - self.mean) / (self.std + self.eps)

    def scaleup(self, data):
        data = self.align_dtypes_and_devices(data)
        return data * (self.std + self.eps) + self.mean
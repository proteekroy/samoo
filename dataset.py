from torch.utils.data import Dataset


class DatasetPrivacy(Dataset):
    def __init__(self, index_tensor, data_tensor, target_tensor, sensitive_tensor):
        Dataset.__init__(self)
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.sensitive_tensor = sensitive_tensor
        self.index_tensor = index_tensor

    def __getitem__(self, index):
        return self.index_tensor[index], self.data_tensor[index], self.target_tensor[index], self.sensitive_tensor[
            index]

    def __len__(self):
        return self.data_tensor.size(0)


class DatasetOpt(Dataset):
    def __init__(self, index_tensor, x, f, g, g_label=None):
        Dataset.__init__(self)
        assert index_tensor == x.size(0)
        assert x.size(0) == f.size(0)
        assert x.size(0) == g.size(0)
        assert x.size(0) == g_label.size(0)

        self.index_tensor = index_tensor
        self.x = x
        self.f = f
        self.g = g
        self.g_label = g_label

    def __getitem__(self, index):
        return self.index_tensor[index], self.x[index], self.f[index], self.g[index], self.g_label[index]

    def __len__(self):
        return self.x.size(0)


class DatasetFunction(Dataset):
    def __init__(self, index_tensor, x, f):
        Dataset.__init__(self)
        assert index_tensor.size(0) == x.size(0)
        assert x.size(0) == f.size(0)

        self.index_tensor = index_tensor
        self.x = x
        self.f = f

    def __getitem__(self, index):
        return self.index_tensor[index], self.x[index], self.f[index]

    def __len__(self):
        return self.x.size(0)


class DatasetConstraint(Dataset):
    def __init__(self, index_tensor, x, g, g_label=None):
        Dataset.__init__(self)
        assert index_tensor.size(0) == x.size(0)
        assert x.size(0) == g.size(0)
        assert x.size(0) == g_label.size(0)

        self.index_tensor = index_tensor
        self.x = x
        self.g = g
        self.g_label = g_label

    def __getitem__(self, index):
        return self.index_tensor[index], self.x[index], self.g[index], self.g_label[index]

    def __len__(self):
        return self.x.size(0)
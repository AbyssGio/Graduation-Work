from jittor import dataset


class Jdataset(dataset.Dataset):
    def __init__(self):
        super().__init__(self)
        self.batchs_list = []
        self.batch_length = []
        self.numofbatchs = 0

    def __getitem__(self, index):
        for batchind in range(0, self.numofbatchs):
            if index > batchind:
                index = index - batchind
                continue
            else:
                return self.batchs_list.__getitem__(index)
        raise AssertionError("Out of index")

    def ConcatDataset(self, datasets):
        for datasetx in datasets:
            self.numofbatchs += 1
            self.batchs_list.append(datasetx)
            self.batch_length.append(len(datasetx))

        return self

    def DataLoader(self,
                   batch_size=512,
                   shuffle=True,
                   pin_memory=True,
                   num_workers=1,
                   drop_last=True
                   ):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last

        return self

    def __batch_len__(self):
        len = 0
        for lens in self.batch_length:
            len += lens
        return len
from .negative_samplers import negative_sampler_factory

from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        save_folder = dataset._get_preprocessed_folder_path()
        print(save_folder)
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.smap = dataset['smap']
        self.user_count = len(self.val)
        self.item_count = len(self.smap)

        if args.test_negative_sample_size >= 0:
            code = args.test_negative_sampler_code
            test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                            self.user_count, self.item_count,
                                                            args.test_negative_sample_size,
                                                            args.test_negative_sampling_seed,
                                                            save_folder)
            self.test_negative_samples = test_negative_sampler.get_negative_samples()
        else:
            self.test_negative_samples =  None
        

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass

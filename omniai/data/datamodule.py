import abc


class DataModule(abc.ABC):
    @abc.abstractmethod
    def get_train_dataloader(self):
        pass

    @abc.abstractmethod
    def get_val_dataloader(self):
        pass

    @abc.abstractmethod
    def get_test_dataloader(self):
        pass

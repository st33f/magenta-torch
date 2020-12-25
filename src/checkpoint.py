import torch
import time
import os
import shutil

class Checkpoint:
    """
    ----------------------------------------------------------------------------
    A Checkpoint class.

    The purpose is to be organized with our experiments, as well to save and
    load progress in training.
    ----------------------------------------------------------------------------
    """

    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'

    def __init__(self, model, epoch, step, optimizer, scheduler,
                samp_rate, KL_rate, free_bits, path=None):
        self.model   = model
        self.epoch     = epoch
        self.step      = step
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.samp_rate = samp_rate
        self.KL_rate   = KL_rate
        self.free_bits = free_bits
        self._path     = path

    @property
    def path(self):
        if self._path is None:
            raise LookupError("This checkpoint has not been saved.")
        return self._path

    def save(self, experiment_dir, model_name=""):

        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        self._path = os.path.join(experiment_dir, self.CHECKPOINT_DIR_NAME, date_time, model_name)
        path = self._path

        #If path exists, erase the whole thing nad make a new one.
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer,
                    'scheduler': self.scheduler,
                    'samp_rate': self.samp_rate,
                    'KL_rate': self.KL_rate,
                    'free_bits': self.free_bits
                   },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))

        return path

    @classmethod
    def load(cls, path):

        #Check if GPU is available.
        if torch.cuda.is_available():
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
            model = torch.load(os.path.join(path, cls.MODEL_NAME))
        else:
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME),map_location=lambda storage, loc: storage)
            model = torch.load(os.path.join(path, cls.MODEL_NAME),map_location=lambda storage, loc: storage)

        #Make RNN parameters contiguous.
#         encoder.flatten_parameters()
#         decoder.flatten_parameters()
        return Checkpoint(model= model,
                        epoch= resume_checkpoint['epoch'],
                        step=resume_checkpoint['step'],
                        optimizer= resume_checkpoint['optimizer'],
                        scheduler = resume_checkpoint['scheduler'],
                        samp_rate = resume_checkpoint['samp_rate'],
                        KL_rate = resume_checkpoint['KL_rate'],
                        free_bits = resume_checkpoint['free_bits'],
                        path= path)
    @classmethod
    def get_latest_checkpoint(cls, exp_path):
        checkpoints_path = os.path.join(exp_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])
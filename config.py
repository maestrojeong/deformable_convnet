SAVE_DIR = './save'

class DeformConConfig(object):
    def __init__(self):
        self.epoch = 100
        self.batch_size = 50
        self.lr = 1e-4
	self.log_every = 1

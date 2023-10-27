class Config(object):
    def __init__(self):
        # system configs
        self.n_nodes = 4
        self.n_meas = 6
        self.ts = 100

        # model configs
        self.embd = 24
        self.n_head = 4
        self.n_layers = 6
        self.ffn_dim = 128
        self.dropout = 0.1

        # training configs
        self.drop_last = False
        self.n_pred = 4
        self.batch_size = 64
        self.epoch = 100
        self.lr = 3e-4

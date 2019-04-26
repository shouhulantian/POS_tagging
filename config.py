class LSTMConfig(object):

    model = 'LSTM'  # the name of used model, in  <models/__init__.py>
    data = 'NYT'  # SEM NYT FilterNYT

    result_dir = './out'

    load_model_path = 'checkpoints/model.pth'  # the trained model

    batch_size = 128  # batch size

    TAG_PATH = 'penn-tree-bank/treebank/tagged'
    RAW_PATH = 'penn-tree-bank/treebank/raw'
    POS_FILE = 'pos_tag.txt'
    WORD_FILE = 'word_en.txt'
    TAG_FILE = 'tag_en.txt'

    num_workers = 0  # how many workers for loading data

    word_dim = 50
    char_dim = 20
    hidden_dim = 50

    norm_emb=True

    num_epochs = 32  # the number of epochs for training
    drop_out = 0.5
    lr = 0.0003  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.0001  # optimizer parameter

    optim = 'SGD'

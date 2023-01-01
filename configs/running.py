class TrainConfig:
    Fs = 30
    record_path = ""
    trans = None
    batch_size = 4
    lr = 1e-3
    num_epochs = 10
    device = "cuda:0"  # cuda:0


class TestConfig:
    Fs = 30
    record_path = ""
    trans = None
    batch_size = 4
    device = "cuda:0"  # cpu

    post = "fft"
    diff = True
    detrend = True


class TrainEfficient:
    Fs = 30
    H = 72
    # no crop
    record_path = ""
    trans = None
    batch_size = 4  # Transformer 2
    lr = 1e-3
    num_epochs = 10
    device = "cuda:3"  # cuda:2
    num_gpu = 1
    frame_depth = 10


class TestEfficient:
    Fs = 30
    # no crop
    record_path = ""
    trans = None
    batch_size = 1
    device = "cuda:3"  # cpu

    post = "peak"
    diff = True
    detrend = True
    num_gpu = 1
    frame_depth = 10

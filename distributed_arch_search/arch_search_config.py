import os


class config:
    host = 'your_host_address'
    username = 'test'
    port = 5672

    exp_name = os.path.dirname(os.path.abspath(__file__))
    exp_name = '-'.join(i for i in exp_name.split(os.path.sep) if i)
    log_dir = 'searchlog/DetNAS_300M_FPN'

    test_send_pipe = exp_name + '-test-send_pipe'
    test_recv_pipe = exp_name + '-test-recv_pipe'

    net_cache = 'model_and_data/checkpoint.pth.tar'

    blocks_keys = [
        'shufflenet_3x3',
        'shufflenet_5x5',
        'shufflenet_7x7',
        'xception_3x3',
    ]
    nr_layer=20

    states=[len(blocks_keys)]*nr_layer

    # flops_limit=None
    flops_limit=300*1e6

    max_epochs=20
    select_num = 10
    population_num = 50
    mutation_num = 20
    m_prob = 0.1
    crossover_num = 20

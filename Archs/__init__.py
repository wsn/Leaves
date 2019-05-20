def create_model(opt):

    if opt['task'] == '1' or '2':
        net = define_net(opt['networks'])
        return net
    else:
        raise NotImplementedError('The task [%s] of networks is not recognized.' % opt['task'])

def define_net(opt):

    which_model = opt['which_model'].upper()
    print('===> Building network [%s]...' % which_model)

    if which_model == 'SIMPLE':
        from .simple_arch import Simple
        net = Simple(opt['num_features'], opt['weight_decay'], opt['initializer'])
    elif which_model == 'NAIVE':
        from .naive_arch import Naive
        net = Naive(opt['num_features'])
    elif which_model == 'MODEL_RESNET':
        from .Model_resnet_arch import Model_resnet
        net = Model_resnet(opt['img_channels'],opt['width'],opt['height'])
    elif which_model == 'MODEL_FCN8':
        from .Model_fcn8_arch import Model_fcn8
        from .Model_fcn8_arch import Model_fcn32
        net = Model_fcn8(opt['num_classes'], opt['num_features'])
    elif which_model == 'MODEL_FCN32':
        from .Model_fcn8_arch import Model_fcn32
        net = Model_fcn32(opt['num_classes'], opt['num_features'])
    else:
        raise NotImplementedError('Network [%s] is not recognized.' % which_model)

    return net

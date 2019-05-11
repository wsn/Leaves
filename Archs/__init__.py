def create_model(opt):

	if opt['task'] == '1' or '2':
		net = define_net(opt['network'])
		return net
	else:
		raise NotImplementedError('The task [%s] of networks is not recognized.' % opt['task'])

def define_net(opt):

	which_model = opt['which_model'].upper()
	print('===> Building network [%s]...' % which_model)

	if which_model == 'SIMPLE':
		from .simple_arch import Simple
		net = Simple(opt['num_features'])
	else:
		raise NotImplementedError('Network [%s] is not recognized.' % which_model)

	return net
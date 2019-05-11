from .solver1 import Solver1

def create_solver(opt):
    if opt['task'] == '1':
        solver = Solver1(opt)
    elif opt['task'] == '2':
        pass
    else:
        raise NotImplementedError('The task [%s] of networks is not recognized.' % opt['task'])

    return solver
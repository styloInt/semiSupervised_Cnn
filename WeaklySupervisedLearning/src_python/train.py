path_caffe = '/home/atemmar/caffe/';

import sys

sys.path.insert(0, path_caffe + '/python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

solver_file = sys.argv[1]
nb_step = int(sys.argv[2])


solver = caffe.SGDSolver(solver_file)

# weight_file = None
# if len(sys.argv) > 3:
# 	weight_file = sys.argv[3]
	# solver.net.copy_from(weight_file)

solver_file = None
if len(sys.argv) > 3:
	solver_state = sys.argv[3]
	solver.restore(solver_state)

solver.step(nb_step)




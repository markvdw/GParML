from pylab import *
import pickle
import numpy

NUM_COLORS = 100
cm = get_cmap('gist_rainbow')
color = []
for i in range(NUM_COLORS):
    color += [cm(1.*i/NUM_COLORS)]  # color will now be an RGBA tuple



path = '/easydata1k/tmp/'

with open('easydata1k/tmp/time_acc.obj','rb') as f:
    acc = pickle.load(f)

with open('easydata1k/tmp/time_acc_SCG_adapted.obj','rb') as f:
    acc_opt = pickle.load(f)

''' Global functions '''

plot(acc['time_acc_statistics_map_reduce'])
title('time_acc_statistics_map_reduce')
ylim(0,2); xlabel('iter'); ylabel('time (s)')
show()

plot(acc['time_acc_calculate_global_statistics'])
title('time_acc_calculate_global_statistics')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()

plot(acc['time_acc_embeddings_MR'])
title('time_acc_embeddings_MR')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()


''' Global functions - time per node '''

y = numpy.array(acc['time_acc_statistics_mapper']).T
x = numpy.arange(y.shape[1])
y_stack = numpy.cumsum(y, axis=0)   # a 3x10 array
fig = figure()
ax1 = fig.add_subplot(111)
ax1.fill_between(x, 0, y_stack[0,:], color=color[numpy.random.randint(NUM_COLORS)], alpha=.7)
for i in xrange(1,y.shape[0]):
    ax1.fill_between(x, y_stack[i-1,:], y_stack[i,:], color=color[numpy.random.randint(NUM_COLORS)], alpha=.7)

title('time_acc_statistics_mapper')
show()




y = numpy.array(acc['time_acc_statistics_reducer']).T
x = numpy.arange(y.shape[1])
y_stack = numpy.cumsum(y, axis=0)   # a 3x10 array
fig = figure()
ax1 = fig.add_subplot(111)
ax1.fill_between(x, 0, y_stack[0,:], color=color[numpy.random.randint(NUM_COLORS)], alpha=.7)
for i in xrange(1,y.shape[0]):
    ax1.fill_between(x, y_stack[i-1,:], y_stack[i,:], color=color[numpy.random.randint(NUM_COLORS)], alpha=.7)

title('time_acc_statistics_reducer')
show()





y = numpy.array(acc['time_acc_embeddings_MR_mapper']).T
x = numpy.arange(y.shape[1])
y_stack = numpy.cumsum(y, axis=0)   # a 3x10 array
fig = figure()
ax1 = fig.add_subplot(111)
ax1.fill_between(x, 0, y_stack[0,:], color=color[numpy.random.randint(NUM_COLORS)], alpha=.7)
for i in xrange(1,y.shape[0]):
    ax1.fill_between(x, y_stack[i-1,:], y_stack[i,:], color=color[numpy.random.randint(NUM_COLORS)], alpha=.7)

title('time_acc_embeddings_MR_mapper')
show()


''' Optimiser functions '''

plot(acc_opt['embeddings_set_grads'])
title('embeddings_set_grads')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()

plot(acc_opt['embeddings_get_grads_mu'])
title('embeddings_get_grads_mu')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()

plot(acc_opt['embeddings_get_grads_kappa'])
title('embeddings_get_grads_kappa')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()

plot(acc_opt['embeddings_get_grads_theta'])
title('embeddings_get_grads_theta')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()

plot(acc_opt['embeddings_get_grads_current_grad'])
title('embeddings_get_grads_current_grad')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()

plot(acc_opt['embeddings_get_grads_gamma'])
title('embeddings_get_grads_gamma')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()

plot(acc_opt['embeddings_get_grads_max_d'])
title('embeddings_get_grads_max_d')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()

plot(acc_opt['embeddings_set_grads_update_d'])
title('embeddings_set_grads_update_d')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()

plot(acc_opt['embeddings_set_grads_update_X'])
title('embeddings_set_grads_update_X')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()

plot(acc_opt['embeddings_set_grads_update_grad_old'])
title('embeddings_set_grads_update_grad_old')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()

plot(acc_opt['embeddings_set_grads_update_grad_new'])
title('embeddings_set_grads_update_grad_new')
ylim(0,1); xlabel('iter'); ylabel('time (s)')
show()


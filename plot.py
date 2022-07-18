import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from utils_git.utils_plot import plot_hyperparams_vs_loss_multiple_algos
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import numpy as np
import os
import datetime

home_path = args['home_path']
name_dataset = args['dataset']
args['if_contour'] = True

algorithm = ''

list_lr = [1e-07, 3e-07, 1e-06, 3e-06, 1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003]
args['list_damping'] = [1e-05, 3e-05, 0.0001, 0.0002, 0.0003, 0.001, 0.003, 0.01]

args['N1'] = 1000
args['N2'] = 1000

N1 = args['N1']
N2 = args['N2']

fetched_data = []

for lr in list_lr:
    working_dir = home_path + 'result/' + name_dataset + '/' + algorithm + '/' + 'if_gpu_True/' +\
              'alpha_' + str(lr) + '/' +\
              'N1_' + str(N1) + '/' +\
              'N2_' + str(N2) + '/'

    os.chdir(working_dir)
    list_file = os.listdir()

    for file_ in list_file:
        with open(file_, 'rb') as fp:
            data_ = pickle.load(fp)
        if algorithm in ['Kron-BFGS-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad',
                         'Kron-BFGS(L)-homo-no-norm-gate-HessianAction-momentum-s-y-Powell-double-damping-regularized-grad-momentum-grad']:
            damping_keyword = 'Kron_BFGS_A_LM_epsilon'
        elif algorithm == 'kfac-no-max-no-LM-momentum-grad':
            damping_keyword = 'kfac_damping_lambda'
        elif algorithm == 'MBNGD-all-to-one-Avg-LRdecay':
            damping_keyword = 'mbngd_damping_lambda'
        else:
            print('error: unknown algorithm for ' + algorithm)
            sys.exit()
            
        if data_['params'][damping_keyword] in args['list_damping']:
            fetched_data.append([lr, data_['params'][damping_keyword], 
                                 np.min(data_['train_losses'])])

fetched_data = np.asarray(fetched_data)
hyperparams = fetched_data #: grid of results stored as a list of (lr, damping, value)

x = []
y = []
z = []

for x_t in hyperparams:
    x += [x_t[0]]
    y += [x_t[1]]
    z += [x_t[2]]

step = 5

if args['dataset'] in ['CURVES-autoencoder-relu-sum-loss',
                       'MNIST-autoencoder-relu-N1-1000-sum-loss']:
    min_level = 50
    max_level = 200
elif args['dataset'] == 'FacesMartens-autoencoder-relu-no-regularization':
    min_level = 5
    max_level = 100
else:
    min_level = 50
    max_level = 200

levels = step*np.arange(min_level/step, max_level/step +1)
xs,ys = np.meshgrid(np.logspace(np.log10(min(x)), np.log10(max(x)), num=25),
                    np.logspace(np.log10(min(y)), np.log10(max(y)), num=25))

resampled = griddata((x, y), z, (xs, ys), method = 'linear')
plt.rcParams.update({'font.size': 18})
plt.rc('font', family='serif')
plt.style.use('seaborn-muted')

fig = plt.figure(figsize = (12,10))
cp = plt.contourf(xs, ys, resampled, levels = levels, cmap = cm.RdYlBu,  extend='max')
fig.patch.set_facecolor('white')


plt.xscale('log')
plt.yscale('log')

plt.colorbar(cp)

plt.xlabel('learning rate')
plt.ylabel('damping')

saving_dir = 

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

plt.savefig(saving_dir +str(datetime.datetime.now().strftime('%Y-%m-%d-%X')) + '.pdf', 
            bbox_inches='tight')
plt.show()

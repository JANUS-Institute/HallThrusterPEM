import torch
import torch.nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# import matplotlib; matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import sys
from scipy.optimize import direct
from scipy.interpolate import BarycentricInterpolator
import pickle

sys.path.append('..')
from utils import ax_default


class Feedforward(torch.nn.Module):

    def __init__(self, layers):
        """Construct a feedforward neural network
        :param layers: A list() of the form [Nx, N1, N2, ..., Ny], where Ni specifies the number of nodes
                        in the ith layer. Nx=input dimension, Ny=output dimension
        """
        super(Feedforward, self).__init__()
        self.layer_counts = layers.copy()
        torch_layers = []
        for k in range(len(layers)-1):
            torch_layers.append(torch.nn.Linear(layers[k], layers[k+1]))

            # Activation function for all but last layer
            if k < len(layers)-2:
                torch_layers.append(torch.nn.Tanh())

        self.ff_stack = torch.nn.Sequential(*torch_layers)
        self.loss_func = torch.nn.MSELoss()
        self.opt = None
        self.iteration = 0
        self.input_settings = {}    # minmax norm settings for the input
        self.output_settings = {}   # minmax norm settings for the output

    def fix_input_shape(self, x):
        x = np.atleast_1d(x).astype(np.float32)
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return torch.tensor(x)

    def extract_params(self):
        """Assumes a layer structure of [1, 2, 2, 1]"""
        params = np.zeros((2, 7), dtype=np.float32)
        params[:, 0] = self.ff_stack[0].weight.data.detach().numpy().squeeze()
        params[:, 1] = self.ff_stack[0].bias.data.detach().numpy().squeeze()
        params[:, 2:4] = self.ff_stack[2].weight.data.detach().numpy()
        params[:, 4] = self.ff_stack[2].bias.data.detach().numpy().squeeze()
        params[:, 5] = self.ff_stack[4].weight.data.detach().numpy().squeeze()
        params[0, 6] = self.ff_stack[4].bias.data.detach().numpy().squeeze()

        return params

    def load_params(self, params):
        """Assumes a layer structure of [1, 2, 2, 1]
        :param params: (2, 7) array of 2x1 weights and biases for each layer
        """
        with torch.no_grad():
            self.ff_stack[0].weight[:] = torch.from_numpy(params[:, 0]).reshape(self.ff_stack[0].weight.shape)
            self.ff_stack[0].bias[:] = torch.from_numpy(params[:, 1]).reshape(self.ff_stack[0].bias.shape)
            self.ff_stack[2].weight[:] = torch.from_numpy(params[:, 2:4]).reshape(self.ff_stack[2].weight.shape)
            self.ff_stack[2].bias[:] = torch.from_numpy(params[:, 4]).reshape(self.ff_stack[2].bias.shape)
            self.ff_stack[4].weight[:] = torch.from_numpy(params[:, 5]).reshape(self.ff_stack[4].weight.shape)
            self.ff_stack[4].bias[:] = torch.from_numpy(np.atleast_1d(params[0, 6])).reshape(self.ff_stack[4].bias.shape)

    def num_params(self):
        Nt = 0
        for k in range(1, len(self.layer_counts)):
            Nt = Nt + self.layer_counts[k-1]*self.layer_counts[k] + self.layer_counts[k]

        return Nt

    def mapminmax(self, x, direction):
        """
        Pre and post process normalization for FNN. Replicates Matlab's mapminmax function.
        xdim=number of inputs, ydim=number of outputs, Nx=number of samples
        :param x: (Nx, xdim) if 'Forward', (Nx, ydim) if 'Reverse', data that is to be normalized
        :param direction: 'Forward' for network inputs, 'Reverse' for network outputs
        --returns--
            xnorm: Normalized inputs if 'Forward'
            y: Un-normalized outputs if 'Reverse'
        """
        # Unpack process settings
        ps = self.input_settings if direction == 'Forward' else self.output_settings
        if len(ps) == 0:
            raise Exception('Normalization settings have not been set yet. Must call model.fit() first')
        xmin = ps['xmin']   # (xdim or ydim,)
        xmax = ps['xmax']   # (xdim or ydim,)
        ymin = ps['ymin']   # scalar (same for all y)
        ymax = ps['ymax']   # scalar (same for all y)

        if direction == 'Forward':
            # Apply the mapping x -> xnorm
            xnorm = ((x - xmin) / (xmax - xmin)) * (ymax - ymin) + ymin
            return xnorm
        elif direction == 'Reverse':
            # Undo the mapping y -> ynorm (equivalently, apply the mapping ynorm -> y)
            ynorm = x  # raw network output is already normalized
            ynorm_min = ymin
            ynorm_max = ymax
            y = ((ynorm - ynorm_min) / (ynorm_max - ynorm_min)) * (xmax - xmin) + xmin
            return y

    def get_data_loaders(self, xdata, ydata, split, bs):
        """Break into training and validation sets and return torch data loaders"""
        xdata = self.fix_input_shape(xdata)
        ydata = self.fix_input_shape(ydata)
        Nx = xdata.size()[0]
        idx = np.arange(0, Nx)
        np.random.shuffle(idx)
        Ntrain = int(np.floor(split * Nx))
        xtrain = xdata[idx[0:Ntrain], ...]
        ytrain = ydata[idx[0:Ntrain], ...]
        xval = xdata[idx[Ntrain:], ...]
        yval = ydata[idx[Ntrain:], ...]
        train_ds = TensorDataset(xtrain, ytrain)
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_ds = TensorDataset(xval, yval)
        val_dl = DataLoader(val_ds, batch_size=None)

        return train_dl, val_dl

    def norm_settings(self, xdata, ydata):
        """Save settings needed for normalization
        :param xdata: training locations (Nx, xdim)
        :param ydata: function value at training locations (Nx, ydim)
        """
        # Input normalization
        self.input_settings['xmin'] = np.min(xdata, axis=0)
        self.input_settings['xmax'] = np.max(xdata, axis=0)
        self.input_settings['ymin'] = -1
        self.input_settings['ymax'] = 1

        # Output normalization
        self.output_settings['xmin'] = np.min(ydata, axis=0)
        self.output_settings['xmax'] = np.max(ydata, axis=0)
        self.output_settings['ymin'] = -1
        self.output_settings['ymax'] = 1

    @staticmethod
    def moving_average(x, N):
        cumsum = np.cumsum(x)
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def forward(self, x):
        x = self.fix_input_shape(x)
        xnorm = self.mapminmax(x, 'Forward')
        ynorm = self.ff_stack(xnorm)
        y = self.mapminmax(ynorm, 'Reverse')
        return y

    def fit(self, xdata, ydata, split=0.8, num_epochs=100, bs=32, animate=False, plot=True, verbose=True, tol=1e-5):
        """Train the NN
        :param xdata: training locations (Nx, xdim)
        :param ydata: function value at training locations (Nx, ydim)
        :param split: training/validation split fraction
        :param num_epochs: number of training iterations
        :param bs: batch size for each gradient step
        :param animate: whether to include animated plot of training loss
        :param plot: whether to plot training results
        :param verbose: whether to print out validation MSE during training
        :param tol: absolute tolerance in validation MSE to stop training
        """
        # Set normalization process settings
        self.norm_settings(xdata, ydata)

        # Get train/val data and set optimizer
        train_dl, val_dl = self.get_data_loaders(xdata, ydata, split, bs)
        self.opt = torch.optim.Adam(self.parameters())
        iters_per_epoch = len(train_dl)

        # Allocate space
        self.iteration = 1
        iters = []
        epochs = []
        train_loss = []
        val_loss = []
        if plot:
            fig, ax = plt.subplots()
            ax.grid()

        # Setup SGD optimization loop
        def opt_loop(epoch):
            self.train()
            for xb, yb in train_dl:
                # Compute loss on mini-batch
                loss = self.loss_func(self(xb), yb)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                iters.append(self.iteration)
                train_loss.append(float(loss))
                self.iteration += 1

                # Handle plotting
                if plot:
                    if animate:
                        ax.clear()
                        ax.grid()
                        if float(loss) < 1:
                            ax.set_yscale('log')
                        ax.plot(iters, train_loss, '-b', label='Train')
                        if len(epochs) > 0:
                            ax.plot(epochs, val_loss, '-r', label='Validation')
                        ax_default(ax, 'Iteration', 'MSE', legend=True)

            # Compute validation loss at each epoch
            self.eval()
            with torch.no_grad():
                loss = np.mean([float(self.loss_func(self(xb), yb)) for xb, yb in val_dl])
                epochs.append((epoch + 1) * iters_per_epoch)
                val_loss.append(float(loss))

            if plot:
                if animate:
                    ax.clear()
                    ax.grid()
                    if float(loss) < 1:
                        ax.set_yscale('log')
                    ax.plot(iters, train_loss, '-b', label='Train')
                    ax.plot(epochs, val_loss, '-r', label='Validation')
                    ax_default(ax, 'Iteration', 'MSE', legend=True)

            if verbose:
                print(f'Epoch: {epoch}, Validation MSE: {loss}')

            # Compute whether we should end training (use a moving average with window of size 10)
            window_size = 10
            if len(val_loss) > 2 * window_size:
                last_val_loss = self.moving_average(np.atleast_1d(val_loss), window_size)[-1]
                end_training = last_val_loss < tol
            else:
                end_training = False
            return end_training

        # Run SGD optimization loop
        if plot and animate:
            ani = FuncAnimation(fig, opt_loop, frames=num_epochs, interval=1, repeat=False)
        else:
            for epoch in range(num_epochs):
                training_end_reached = opt_loop(epoch)
                if training_end_reached:
                    break
            if plot:
                ax.plot(iters, train_loss, '-b', label='Train')
                ax.plot(epochs, val_loss, '-r', label='Validation')
                ax.set_yscale('log')
                ax_default(ax, 'Iteration', 'MSE', legend=True)

        if plot:
            plt.show()


def leja_1d(N, z_bds, z_pts=None, wt_fcn=None):
    """Find the next N points in the leja sequence of z_pts
    :param N: Number of new points to add to the sequence
    :param z_bds: Bounds on 1d domain (tuple)
    :param z_pts: Current univariate leja sequence (Nz,), start at middle of z_bds if None
    :param wt_fcn: Weighting function, uses a constant weight if None, callable as wt_fcn(z)
    """
    if wt_fcn is None:
        wt_fcn = lambda z: 1
    if z_pts is None:
        z_pts = (z_bds[1] + z_bds[0]) / 2
        N = N - 1
    z_pts = np.atleast_1d(z_pts)

    for i in range(N):
        obj_fun = lambda z: -wt_fcn(z)*np.prod(np.abs(z-z_pts))
        res = direct(obj_fun, [z_bds])  # Use global DIRECT optimization over 1d domain
        z_star = res.x

        # Plot objective function (for testing)
        # fig, ax = plt.subplots()
        # Ng = 100
        # Nz = z_pts.shape[0]
        # z_grid = np.linspace(z_bds[0], z_bds[1], Ng)
        # diff = z_grid.reshape((Ng, 1)) - z_pts.reshape((1, Nz))  # (Ng, Nz)
        # func = wt_fcn(z_grid) * np.prod(np.abs(diff), axis=1)  # (Ng,)
        # ax.plot(z_grid, func, '-k', label='Objective')
        # ax.plot(z_star, -obj_fun(z_star), '*r', markersize=10, label=r'$z^*$')
        # ax_default(ax, xlabel='z', ylabel='f(z)', legend=True)
        # plt.show()

        z_pts = np.concatenate((z_pts, z_star))

    return z_pts


def chebyshev_1d(N, z_bds):
    k = np.arange(1, N+1)
    return (1/2)*(z_bds[1] + z_bds[0]) + (1/2)*(z_bds[1] - z_bds[0])*np.cos((2*k-1)/(2*N)*np.pi)


# Black-box test function to fit (simple tanh)
def bb_func(x, A=2, L=1, frac=4):
    return A*np.tanh(2/(L/frac)*(x-L/2)) + A


def test_interp():
    """Test leja/chebyshev barycentric Lagrange interpolation"""
    # Hyper-parameters
    layers = [1, 2, 2, 1]
    num_epochs = 1500
    bs = 32
    split = 0.8

    # Load data
    xdata = np.linspace(0, 1, 100)
    ydata = bb_func(xdata)
    # with open('ff_mc_0.json', 'r') as fd:
    #     data = json.load(fd)
    #     xdata = np.array(data['model1']['output']['z'])
    #     ydata = np.array(data['model1']['output']['ui_1'])

    # Generate 1d leja/chebyshev sequences
    z0 = np.array([0.2])
    z_bds = (0, 1)
    N = 5
    z_leja = leja_1d(N, z0, z_bds)
    y_leja = bb_func(z_leja)
    z_cheb = chebyshev_1d(N, z_bds)
    y_cheb = bb_func(z_cheb)

    # Barycentric lagrange interpolation
    interp_leja = BarycentricInterpolator(z_leja, y_leja)
    y_interp_leja = interp_leja(xdata)
    interp_cheb = BarycentricInterpolator(z_cheb, y_cheb)
    y_interp_cheb = interp_cheb(xdata)

    # Neural network interpolation
    # model_leja = Feedforward(layers)
    # model_leja.fit(z_leja, y_leja, num_epochs=num_epochs, bs=bs, split=split, animate=False)
    # y_nn_leja = model_leja(xdata).detach().numpy().squeeze()
    # model_cheb = Feedforward(layers)
    # model_cheb.fit(z_cheb, y_cheb, num_epochs=num_epochs, bs=bs, split=split, animate=False)
    # y_nn_cheb = model_cheb(xdata).detach().numpy().squeeze()

    # Plot model fit
    fig, ax = plt.subplots()
    ax.plot(xdata, ydata, '-k', label=r'Truth')
    ax.plot(z_leja, y_leja, 'or', label=r'$x_{leja}$')
    ax.plot(xdata, y_interp_leja, '-r', label=r'Leja')
    # ax.plot(xdata, y_nn_leja, '--r', label=r'FNN leja')
    ax.plot(z_cheb, y_cheb, 'ob', label=r'$x_{cheby}$')
    ax.plot(xdata, y_interp_cheb, '-b', label=r'Chebyshev')
    # ax.plot(xdata, y_nn_cheb, '--b', label=r'FNN cheby')
    ax_default(ax, 'x', 'f(x)', legend=True)
    plt.show()


def fit_ff():
    """Test variance in NN parameters for various fits"""
    N = 10
    bds = (2, 8)
    x = chebyshev_1d(N, bds)    # x is an input parameter to the bb_func
    x_grid = np.linspace(bds[0], bds[1], 100)
    z = np.linspace(0, 1, 100)  # z is the axial profile for the NN to fit

    # Hyperparameters for the NN
    epochs = 2000
    layers = [1, 2, 2, 1]
    bs = 32
    split = 0.8

    # Loop over inputs x and train NN on each
    nn_params = np.zeros((N, 2, 7))
    model = Feedforward(layers)
    for i in range(N):
        # Generate training data and fit
        y = bb_func(z, frac=x[i])
        model.fit(z, y, split=split, num_epochs=epochs, bs=bs, animate=False, tol=1e-5)
        params = model.extract_params()
        nn_params[i, :, :] = params

    # Interpolate and plot NN parameters against x
    interp = BarycentricInterpolator(x)
    fig, ax = plt.subplots(2, 7, sharex='col')
    for i in range(2):
        for j in range(7):
            interp.set_yi(nn_params[:, i, j])
            y_interp = interp(x_grid)
            ax[i, j].plot(x, nn_params[:, i, j], 'ob')
            ax[i, j].plot(x_grid, y_interp, '-b')
            ax_default(ax[i, j], '', '', legend=False)

            if j == 0:
                ax[i, j].set_ylabel('NN parameter ($w$ or $b$)')
            if i == 1:
                ax[i, j].set_xlabel('Model hyperparameter $h$')

    fig.set_size_inches(21, 6)
    fig.tight_layout()
    plt.show()

    # Save to file
    with open('nn_interp.pkl', 'wb') as fd:
        save_dict = {'x': x, 'nn_params': nn_params, 'x_bds': bds}
        pickle.dump(save_dict, fd)


def test_ff():
    # Load interpolation params
    with open('nn_interp.pkl', 'rb') as fd:
        data = pickle.load(fd)
    x = data['x']
    nn_params = data['nn_params']
    x_bds = data['x_bds']
    z_grid = np.linspace(0, 1, 100)
    interp = BarycentricInterpolator(x)

    # Generate test set
    N_test = 5
    x_test = np.random.rand(N_test)*(x_bds[1] - x_bds[0]) + x_bds[0]
    nn_test = np.zeros((N_test, 2, 7))
    model = Feedforward([1, 2, 2, 1])

    # Interpolate NN parameters at each test case
    for j in range(2):
        for k in range(7):
            interp.set_yi(nn_params[:, j, k])
            nn_test[:, j, k] = interp(x_test)

    # Compute NN prediction for each test case
    fig, ax = plt.subplots(1, N_test)
    for i in range(N_test):
        y_truth = bb_func(z_grid, frac=x_test[i])
        model.norm_settings(z_grid, y_truth)        # Normalize on truth for now, later will need to predict bounds)
        model.load_params(nn_test[i, :, :])
        y_pred = model(z_grid).detach().numpy()
        ax[i].plot(z_grid, y_truth, '-k', label='Truth')
        ax[i].plot(z_grid, y_pred, '--r', label='NN')
        ax[i].set_title(f'h={x_test[i]:.2f}')
        ax_default(ax[i], 'x', 'f(x)', legend=True)
    fig.set_size_inches(3*N_test, 3)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # test_interp()     # Test interpolation schemes
    # plot_bb()         # Plot black-box function
    # fit_ff()          # Fit NN over range of hyper-parameters
    test_ff()           # Test interpolation of NN params to new hyper-parameters

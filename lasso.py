import argparse
import numpy as np
import jax.numpy as jnp
from numpy.linalg import eigh
from jax import grad, jit, vmap
from scipy.io import loadmat
import matplotlib.pyplot as plt


def parse_command():
    parser = argparse.ArgumentParser('Porximal Gradient for Lasso')
    parser.add_argument('--data_path', default='./data/QuizeData.mat', type=str, help='path to the dataset')
    parser.add_argument('--Lambda', default=0.2, type=float, help='hyperparameter of regularization')
    parser.add_argument('--tol', default=1e-8, type=float, help='absoluate tolerance')
    parser.add_argument('--max_iter', default=1000, type=int, help='maximum iterations')
    parser.add_argument('--silent', default=False, action='store_true', help='flag for printing opt process')
    args = parser.parse_args()

    return args


def plot_loss(loss):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(np.arange(len(loss))+1, loss, label='loss')
    
    ax.legend()
    ax.set_xlabel('#step', fontsize=10)
    ax.set_ylabel(r'$F(x)$', fontsize=10)
    ax.grid(linestyle=':')
    fig.tight_layout()
    plt.savefig("./loss_curve.pdf", dpi=200)
    plt.close()


class Lasso(object):

    def __init__(self, A, b, _lambda):
        super(Lasso, self).__init__()
        
        # initialize values
        self.A, self.b = A, b
        self.n, self.p = A.shape
        self._lambda = _lambda

        # calculate the Lipschitz constant
        eig_vals, _ = eigh(jnp.dot(A.T, A))
        self.L_f = eig_vals[-1]  

        # initialize params
        self.x = jnp.zeros(self.p)
        self.grad_f = jit(grad(self.feval))
    
    def reset(self):
        self.x = jnp.zeros(self.p)
    
    def feval(self, x):
        fx = 0.5 * jnp.sum((jnp.dot(self.A, x) - self.b)**2)
        return fx
    
    def objective(self):
        fx = self.feval(self.x)
        gx = self._lambda * jnp.sum(jnp.abs(self.x))
        return fx + gx
    
    def prox_g(self, v, thres):
        return jnp.maximum(0, v-thres) - jnp.maximum(0, -v-thres)
    
    def grad(self):
        self.grad_x = self.grad_f(self.x)
    
    def step(self):
        self.x = self.prox_g(self.x - (1./self.L_f) * self.grad_x, self._lambda*(1./self.L_f))


if __name__ == "__main__":
    args = parse_command()

    # read data
    data = loadmat(args.data_path)['lasso']
    A = jnp.reshape(data[0,0][0], data[0,0][0].shape)
    b = jnp.reshape(data[0,0][1], (-1,))
    x_org = jnp.reshape(data[0,0][2], (-1,))

    # initilize lasso model
    model = Lasso(A, b, args.Lambda)
    
    # training loop
    prox_optval = []
    for i in range(args.max_iter):
        # back propogation
        model.grad()

        # update parameters
        model.step()
        loss = model.objective()
        
        if not args.silent:
            print("step: {:3d}, loss: {:7.4f}, ||Ax* - b||_2^2: {:6.4f}, ||x* - x||_0: {:3d}".format(
                i+1, loss, model.feval(model.x), jnp.count_nonzero(x_org - model.x)))
        prox_optval.append(loss)

        if i > 1 and np.abs(prox_optval[-1] - prox_optval[-2]) < args.tol:
            break
    
    print("-"*40)
    print('The final result on lambda={} is:'.format(args.Lambda))
    print('||Ax* - b||_2^2: {:6.3f}, ||x* - x||_0: {:2d}'.format(
        model.feval(model.x), jnp.count_nonzero(x_org - model.x)))
    print("nnz of x*: ", jnp.count_nonzero(model.x))
    print("nnz of x_org: ", jnp.count_nonzero(x_org))
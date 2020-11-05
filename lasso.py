import os
import argparse
import numpy as np
import jax.numpy as jnp
from numpy.linalg import eigh
from jax import grad, jit, vmap
from jax.numpy.linalg import inv
from scipy.io import loadmat
import matplotlib.pyplot as plt


def parse_command():
    parser = argparse.ArgumentParser('Proximal Solvers for Penalized Lasso')
    parser.add_argument('--A', default='data/A.mat', type=str, help='path to matrix A')
    parser.add_argument('--b', default='data/b.mat', type=str, help='path to vector b')
    parser.add_argument('--gamma', default=0.3, type=float, help='hyperparameter of regularization')
    parser.add_argument('--opt', default='ISTA', type=str, choices=['ISTA', 'FISTA', 'ADMM'], help='optimizer for lasso')
    parser.add_argument('--tol', default=1e-6, type=float, help='absoluate tolerance')
    parser.add_argument('--max_iter', default=1000, type=int, help='maximum iterations')
    parser.add_argument('--silent', default=False, action='store_true', help='flag for printing opt process')
    parser.add_argument('--output_dir', default='./output', type=str, help='output dir')
    args = parser.parse_args()

    return args


def plot_loss(loss):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(np.arange(len(loss)) + 1, loss, label='loss')

    ax.legend()
    ax.set_xlabel('#step', fontsize=10)
    ax.set_ylabel(r'$F(x)$', fontsize=10)
    ax.grid(linestyle=':')
    fig.tight_layout()
    plt.savefig("./loss_curve.pdf", dpi=200)
    plt.close()


class Lasso(object):

    def __init__(self, A, b, gamma, optimizer='ISTA'):
        super(Lasso, self).__init__()

        # initialize values
        self.A, self.b = A, b
        self.n, self.p = A.shape
        self.gamma = gamma
        self.optimizer = optimizer

        # cache results
        self.AtA = jnp.dot(A.T, A)
        if optimizer == 'ISTA' or optimizer == 'ADMM':
            self.Atb = jnp.dot(A.T, b)
            self.AtA_inverse = inv(self.AtA + jnp.eye(self.p))

        # calculate the Lipschitz constant
        eig_vals, _ = eigh(self.AtA)
        self.L_f = eig_vals[-1]
        self.lr = 1. / self.L_f

        # initialize parameters
        self.x = jnp.zeros(self.p)
        self.z = None
        if optimizer == 'FISTA':
            self.y = self.x
            self.t = 1
        elif optimizer == 'ADMM':
            self.z = jnp.zeros(self.p)
            self.u = jnp.zeros(self.p)

        self.grad_f = jit(grad(self.feval))

    def reset(self):
        self.x = jnp.zeros(self.p)

    def feval(self, x):
        fx = 0.5 * jnp.sum((jnp.dot(self.A, x) - self.b)**2)
        return fx

    def objective(self):
        fx = self.feval(self.x)
        if self.z is None:
            gz = self.gamma * jnp.sum(jnp.abs(self.x))
        else:
            gz = self.gamma * jnp.sum(jnp.abs(self.z))
        return fx + gz

    def prox_g(self, v, thres):
        return jnp.maximum(0, v - thres) - jnp.maximum(0, -v - thres)

    def step(self):
        if self.optimizer == 'ISTA':
            # x-update
            self.x = self.prox_g(self.x - self.lr * self.grad_f(self.x), self.gamma * self.lr)
        elif self.optimizer == 'FISTA':
            # x-update
            self.x_prev = self.x
            self.x = self.prox_g(self.y - self.lr * self.grad_f(self.y), self.gamma * self.lr)

            # y-update
            t_k = self.t
            self.t = (1 + jnp.sqrt(1 + 4 * self.t**2)) / 2.
            self.y = self.x + (t_k - 1) / self.t * (self.x - self.x_prev)
        elif self.optimizer == 'ADMM':
            # x-update
            q = self.Atb + self.z - self.u
            self.x = jnp.dot(self.AtA_inverse, q)

            # z-update
            self.z_prev = self.z
            self.z = self.prox_g(self.x + self.u, self.gamma)

            # u-update
            self.u = self.u + self.x - self.z


if __name__ == "__main__":
    args = parse_command()

    # read data
    A = loadmat(args.A)['A']
    b = loadmat(args.b)['b']
    b = jnp.reshape(b, (-1,))

    # initilize lasso model
    model = Lasso(A, b, args.gamma, optimizer=args.opt)

    # training loop
    prox_optval = []
    prox_optval.append(model.objective())

    for i in range(args.max_iter):
        # update parameters
        model.step()
        loss = model.objective()

        if not args.silent:
            print("step: {:3d}, loss: {:7.4f}, ||Ax* - b||_2^2: {:6.4f}".format(
                i + 1, loss, model.feval(model.x)))
        prox_optval.append(loss)

        if i > 1 and np.abs(prox_optval[-1] - prox_optval[-2]) < args.tol:
            break

    print("-" * 40)
    print('Parameters: gamma={}, solver={}'.format(args.gamma, args.opt))
    print('||Ax* - b||_2^2: {:6.3f}, Obj: {:6.3f}'.format(
        model.feval(model.x), prox_optval[-1]))
    if args.opt == 'ADMM':
        print("nnz of x*: ", jnp.count_nonzero(model.z))
    else:
        print("nnz of x*: ", jnp.count_nonzero(model.x))

    print('Writing the results...')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_file = os.path.join(args.output_dir, '{}_{}.npy'.format(args.opt, args.gamma))
    jnp.save(output_file, jnp.array(prox_optval))
    print('Done')

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_command():
    parser = argparse.ArgumentParser('Plot function of Proximal Methods')
    parser.add_argument('--output_dir', default='./output', help='the output directory')
    parser.add_argument('--gamma', default='0.1,0.3,0.6,1.0', type=str, help='hyperparameter of regularization')
    parser.add_argument('--out_fig', default='./output/loss_curve.pdf', type=str, help='output figure')
    args = parser.parse_args()

    return args


def plot_loss(res_dic, ax, gamma):
    _len = []
    for key, v in res_dic.items():
        _len.append(len(v))
    step = min(_len)

    for key, v in res_dic.items():
        ax.plot(np.arange(step), v[:step], label=key)

    ax.legend()
    ax.set_title(r'$\gamma$={}'.format(gamma))
    ax.set_xlabel('#step', fontsize=10)
    ax.set_ylabel(r'$F(x)$', fontsize=10)
    ax.grid(linestyle=':')


if __name__ == "__main__":
    args = parse_command()
    gamma_list = args.gamma.split(',')

    fig = plt.figure(figsize=(8, 6))

    for i, gamma in enumerate(gamma_list):
        res_dic = {}
        res_dic['ISTA'] = np.load(os.path.join(args.output_dir, 'ISTA_{}.npy'.format(gamma)))
        res_dic['FISTA'] = np.load(os.path.join(args.output_dir, 'FISTA_{}.npy'.format(gamma)))
        res_dic['ADMM'] = np.load(os.path.join(args.output_dir, 'ADMM_{}.npy'.format(gamma)))
        res_dic['FASTA'] = np.load(os.path.join(args.output_dir, 'FASTA_{}.npy'.format(gamma)))

        ax = fig.add_subplot(2, 2, i + 1)
        plot_loss(res_dic, ax, gamma)

    fig.tight_layout()
    plt.savefig("./output/loss_curve.pdf", dpi=200)
    plt.close()

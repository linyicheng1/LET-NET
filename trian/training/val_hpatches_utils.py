import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def load_precompute_errors(file):
    errors = torch.load(file)
    return errors


def draw_MMA(errors):
    plt.switch_backend('agg')
    methods = ['hesaff', 'hesaffnet', 'delf', 'superpoint', 'lf-net', 'd2-net-trained', 'd2-net-trained-ms']
    names = ['Hes. Aff. + Root-SIFT', 'HAN + HN++', 'DELF', 'SuperPoint', 'LF-Net', 'D2-Net Trained',
             'D2-Net Trained MS']
    colors = ['red', 'orange', 'olive', 'green', 'blue', 'brown', 'purple']
    linestyles = ['--', '--', '--', '--', '--', '--', '--']

    methods += ['r2d2_WASF_N16.scale-0.3-1', 'disk', ]
    names += ['r2d2', 'disk', ]
    colors += ['silver', 'sandybrown', ]
    linestyles += ['--', '--', ]

    methods += ['ours']
    names += ['ours']
    colors += ['cyan']
    linestyles += ['-']

    n_i = 52
    n_v = 56

    plt_lim = [1, 10]
    plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)

    plt.rc('axes', titlesize=25)
    plt.rc('axes', labelsize=25)

    fig = plt.figure(figsize=(20, 5))
    canvas = FigureCanvas(fig)

    plt.subplot(1, 4, 1)
    for method, name, color, ls in zip(methods, names, colors, linestyles):
        i_err, v_err, _ = errors[method]
        plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng], color=color, ls=ls,
                 linewidth=3, label=name)
    plt.title('Overall')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylabel('MMA')
    plt.ylim([0, 1])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend()

    plt.subplot(1, 4, 2)
    for method, name, color, ls in zip(methods, names, colors, linestyles):
        i_err, v_err, _ = errors[method]
        plt.plot(plt_rng, [i_err[thr] / (n_i * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
    plt.title('Illumination')
    # plt.xlabel('threshold [px]')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylim([0, 1])
    plt.gca().axes.set_yticklabels([])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.subplot(1, 4, 3)
    for method, name, color, ls in zip(methods, names, colors, linestyles):
        i_err, v_err, _ = errors[method]
        plt.plot(plt_rng, [v_err[thr] / (n_v * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
    plt.title('Viewpoint')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylim([0, 1])
    plt.gca().axes.set_yticklabels([])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.subplot(1, 4, 4)

    canvas.draw()  # draw the canvas, cache the renderer
    width, height = canvas.get_width_height()

    # Option 2a: Convert to a NumPy array.
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    plt.close()

    return image[:, 150:1420, :]


if __name__ == '__main__':
    errors = load_precompute_errors('errors.pkl')
    image = draw_MMA(errors)
    plt.imshow(image), plt.show()

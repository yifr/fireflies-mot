import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import sys
sys.path.append("/Users/yonifriedman/Research/Tracking/Fireflies/src/")
from utils import *

def plot_frame(t, states_over_time, gt_xs, gt_ys, gt_blinks, gt_n_fireflies, n_particles, savedir):
    """
    Plot and save a single frame.
    """
    fig, axs = plt.subplots(1, 6, figsize=(36, 8))

    # Set up ground truth plot
    axs[0].set_facecolor("black")
    axs[0].set_xlim(0, 32)
    axs[0].set_ylim(32, 0)
    axs[0].set_title(f"Ground Truth ({gt_n_fireflies}-Firefly)")

    gt_x = gt_xs[t]
    gt_y = gt_ys[t]

    for i in range(gt_n_fireflies):
        if gt_blinks[i] == 1:
            axs[0].scatter(gt_x[i], gt_y[i], color="y", s=200)
        else:
            axs[0].scatter(gt_x[i], gt_y[i], color="r", alpha=0.6, s=200)

    for i in range(gt_n_fireflies):
        if gt_blinks[i] == 1:
            axs[1].scatter(gt_x[i], gt_y[i], color="y", s=200)
    axs[1].set_facecolor("black")
    axs[1].set_xlim(0, 32)
    axs[1].set_ylim(32, 0)
    axs[1].set_title(f"Observations")

    # Set up other firefly particle plots
    states = states_over_time[t]
    blinking_data = {i: ([], []) for i in range(1, 5)}
    non_blinking_data = {i: ([], []) for i in range(1, 5)}

    for i in range(n_particles):
        s = states[i]
        flags = s.flag
        n_fireflies = int(sum(flags))
        blinking = s.value["blinking"][flags]
        x, y = s.value["x"][flags], s.value["y"][flags]

        blinking_data[n_fireflies][0].extend(x[blinking == 1])
        blinking_data[n_fireflies][1].extend(y[blinking == 1])

        non_blinking_data[n_fireflies][0].extend(x[blinking == 0])
        non_blinking_data[n_fireflies][1].extend(y[blinking == 0])

    for n_fireflies in range(1, 5):
        axs[n_fireflies + 1].set_facecolor("black")
        axs[n_fireflies + 1].set_xlim(0, 32)
        axs[n_fireflies + 1].set_ylim(32, 0)
        axs[n_fireflies + 1].set_title(f"{n_fireflies}-Firefly Particles")

        if non_blinking_data[n_fireflies][0]:
            axs[n_fireflies + 1].scatter(
                non_blinking_data[n_fireflies][0],
                non_blinking_data[n_fireflies][1],
                color="r", s=100, alpha=0.3, label="Non-Blinking"
            )
        if blinking_data[n_fireflies][0]:
            axs[n_fireflies + 1].scatter(
                blinking_data[n_fireflies][0],
                blinking_data[n_fireflies][1],
                color="y", marker="o", s=100, label="Blinking"
            )

    plt.suptitle(f"Inference Prediction [t={t}]")
    plt.tight_layout()
    plt.savefig(f"{savedir}/pf_frame_{t}.png")
    plt.close(fig)


def create_smc_animations_parallel(gt_chm, states_over_time, n_particles, savedir="animations/"):
    gt_xs, gt_ys = get_gt_locations(gt_chm)
    gt_n_fireflies = gt_chm["n_fireflies"]

    # Preload ground truth blinking for each time step
    gt_blinks_over_time = gt_chm["steps", :, "dynamics", "blinking", :].value

    os.makedirs(savedir, exist_ok=True)

    # Prepare arguments for each frame
    tasks = [
        (t, states_over_time, gt_xs, gt_ys, gt_blinks_over_time[t], gt_n_fireflies, n_particles, savedir)
        for t in range(len(states_over_time))
    ]

    # Use multiprocessing Pool to parallelize frame plotting
    with Pool() as pool:
        list(tqdm(pool.starmap(plot_frame, tasks), total=len(tasks)))




#####################
# Old and Slow
#####################
def create_smc_animations_serial(gt_chm, states_over_time, n_particles, savedir="animations/"):
    gt_xs, gt_ys = get_gt_locations(gt_chm)
    gt_n_fireflies = gt_chm["n_fireflies"]

    fig, axs = plt.subplots(1, 5, figsize=(30, 8))

    for i, ax in enumerate(axs):
        if i > 0:
            ax.set_title(f"{i}-Firefly Particles")

        ax.set_facecolor("black")
        ax.set_xlim(0, 32)
        ax.set_ylim(32, 0)

    for t in tqdm(range(len(states_over_time))):
        states = states_over_time[t]

        for i, ax in enumerate(axs):
            ax.cla()
            ax.set_xlim(0, 32)
            ax.set_ylim(32, 0)

            if i > 0:
                ax.set_title(f"{i}-Firefly Particles")
            else:
                ax.set_title(f"Ground Truth")

        gt_blinks = gt_chm["steps", :, "dynamics", "blinking", :].value[t]
        gt_x = gt_xs[t]
        gt_y = gt_ys[t]

        for i in range(gt_n_fireflies):
            if gt_blinks[i] == 1:
                axs[0].scatter(gt_x[i], gt_y[i], color="y", s=200)
            else:
                axs[0].scatter(gt_x[i], gt_y[i], color="r", alpha=0.6, s=200)

        axs[0].set_title(f"Ground Truth ({gt_n_fireflies}-Firefly)")

        # Prepare dictionaries for blinking and non-blinking data grouped by firefly count
        blinking_data = {i: ([], []) for i in range(1, 5)}  # x, y per firefly count
        non_blinking_data = {i: ([], []) for i in range(1, 5)}

        # Collect data for each state
        for i in range(n_particles):
            s = states[i]
            flags = s.flag
            n_fireflies = int(np.sum(flags))
            blinking = s.value["blinking"][flags]
            x, y = s.value["x"][flags], s.value["y"][flags]
            
            blinking_x = x[blinking == 1]
            # Append blinking and non-blinking data
            blinking_data[n_fireflies][0].extend(x[blinking == 1])
            blinking_data[n_fireflies][1].extend(y[blinking == 1])

            non_blinking_data[n_fireflies][0].extend(x[blinking == 0])
            non_blinking_data[n_fireflies][1].extend(y[blinking == 0])

        # Plot collected data on corresponding axes
        for n_fireflies in range(1, 5):
            if non_blinking_data[n_fireflies][0]:  # Plot non-blinking fireflies
                axs[n_fireflies].scatter(
                    non_blinking_data[n_fireflies][0],
                    non_blinking_data[n_fireflies][1],
                    color="r",
                    # linestyle="--",
                    # facecolors='none',
                    s=100,
                    alpha=0.3,
                    label="Non-Blinking",
                )

            if blinking_data[n_fireflies][0]:  # Plot blinking fireflies
                axs[n_fireflies].scatter(
                    blinking_data[n_fireflies][0],
                    blinking_data[n_fireflies][1],
                    color="y",
                    marker="o",
                    s=100,
                    label="Blinking",
                )

        plt.suptitle(f"Inference Prediction [t={t}]")
        plt.tight_layout()
        plt.savefig(f"{savedir}/pf_frame_{t}.png")
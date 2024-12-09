import genjax
from genjax import Mask
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import numpy as np

class HiddenIndex:
    def __repr__(self):
        return "#"


class Addr:
    def __init__(self, addr, show_indices):
        if not show_indices:
            new_addr = []
            for a in addr:
                if isinstance(a, str):
                    new_addr.append(a)
                else:
                    new_addr.append(HiddenIndex())
            addr = new_addr
        self.addr = addr

    def __repr__(self):
        return f"<{self.addr}>"

    def __lt__(self, other):
        return self.addr < other.addr


def pytreefy(t, show_indices=False):
    def cm_kv_inner(t, addr_path=None, flag=None):
        if addr_path is None:
            addr_path = []
        else:
            addr_path = addr_path.copy()
        match type(t):
            case genjax._src.core.generative.choice_map.XorChm:
                ret1 = cm_kv_inner(t.c1, addr_path, flag)
                ret2 = cm_kv_inner(t.c2, addr_path, flag)
                # Check for empty intersection
                set1 = set(key.__repr__() for key in ret1.keys())
                set2 = set(key.__repr__() for key in ret2.keys())
                in_common = set1.intersection(set2)
                if not in_common:
                    ret1.update(ret2)
                    return ret1
                else:
                    raise ValueError("Common keys found in XorChm")
            case genjax._src.core.generative.choice_map.OrChm:
                ret1 = cm_kv_inner(t.c1, addr_path, flag)
                ret2 = cm_kv_inner(t.c2, addr_path, flag)
                ret1.update(ret2)
                return ret1
            case genjax._src.core.generative.choice_map.StaticChm:
                addr_path.append(t.addr)
                return cm_kv_inner(t.c, addr_path, flag)
            case genjax._src.core.generative.choice_map.IdxChm:
                addr_path.append(t.addr)
                return cm_kv_inner(t.c, addr_path, flag)
            case genjax._src.core.generative.choice_map.ValueChm:
                if isinstance(t.v, genjax._src.core.generative.choice_map.FilteredChm):
                    return cm_kv_inner(t.v, addr_path, flag)
                # TODO: a better version would replace the masked values with a special symbol indicating masked values
                if flag is None:
                    return {Addr(addr_path, show_indices): t.v}
                else:
                    return {Addr(addr_path, show_indices): (t.v.T * flag.T).T}
            case genjax._src.core.generative.choice_map.MaskChm:
                if flag is None:
                    flag = t.flag
                else:
                    # broadcasting with leading axis on the left
                    flag = (flag.T * t.flag.T).T
                return cm_kv_inner(t.c, addr_path, flag)
            case genjax._src.core.generative.choice_map.EmptyChm:
                return {}
            case genjax._src.core.generative.choice_map.FilteredChm:
                ret = cm_kv_inner(t.c, addr_path)
                # TODO: this should grap the list of addresses not just the top one. this creates a bug in test 6.1
                sel = t.selection.addr
                keys = ret.keys()
                kept_keys = [
                    k for k in keys if all(x == y for x, y in zip(k.addr, sel))
                ]
                return {key: ret[key] for key in kept_keys}
            case _:
                raise NotImplementedError(str(type(t)))

    return cm_kv_inner(t)

def get_masked_values(mask, values=None, fill_value=0.):
    """
    Args:
        mask: Array[Bool] or genjax.Mask: array of masks
        values: jnp.array: If mask isn't a genjax.Mask object 
                            (containing flags and values)
                            values need to be passed explicitly
        fill_value: Any: Some value to fill in masked slots
    """
    if type(mask) == Mask:
        return jnp.where(mask.flag, mask.value, fill_value)
    else:
        return jnp.where(mask, values, fill_value)


def get_frames(chm):
    """
    Expects a model with a pixel observation likelihood
    """
    try:    
        observations = list(chm["steps", ..., "observations", "pixels"].value)
    except:
        raise ValueError('Model does not have choices ["steps", :, "observations", "pixels"]. Double check your observation model')
    
    return observations

def get_observations(chm):
    """
    Args:
        chm: genjax.ChoiceMap: A choice map from a generative model

    Returns:
        observed_xs: Array[Float]: x positions of observed fireflies
        observed_ys: Array[Float]: y positions of observed fireflies
    """
    try:
        observed_xs = chm["steps", :, "observations", "observed_xs"]
        observed_ys = chm["steps", :, "observations", "observed_ys"]

        observed_xs = observed_xs.value
        observed_ys = observed_ys.value
    except:
        raise ValueError('Model does not have choices ["steps", :, "observations", "observed_xs]. Double check your observation model')
    return observed_xs, observed_ys

def get_gt_locations(chm):
    """
    Expects masked dynamics with x and y values. 
    Returns the unmasked x and y values, and -1s in place of masked values
    """
    xs = chm["steps", :, "dynamics", :, "x"]
    ys = chm["steps", :, "dynamics", :, "y"]

    xs = jnp.where(xs.flag, xs.value, -1.)
    ys = jnp.where(ys.flag, ys.value, -1.)
    
    return xs, ys

def get_dynamics(chm, mask_value=-1.):
    """
    Expects masked dynamics with x and y values. 
    """
    xs = chm["steps", :, "dynamics", :, "x"]
    ys = chm["steps", :, "dynamics", :, "y"]
    blinks = chm["steps", :, "dynamics", :, "blink"]
    xs = jnp.where(xs.flag, xs.value, mask_value)
    ys = jnp.where(ys.flag, ys.value, mask_value)
    blinks = jnp.where(blinks.flag, blinks.value, False)
    return xs, ys, blinks

def animate(frames, fps, ax=None):
    if ax == None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    img = ax.imshow(frames[0], vmin=0, vmax=1, cmap="hot")  

    def update(frame):
        img.set_data(frames[frame])  
        ax.set_title(f"Frame {frame}")
        return [img]  

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=True)
    plt.close()
    return ani

def animate_fireflies(x, y, blink,
                     mask_value=-1,  # Value indicating masked positions (0 or -1)
                     fps=30, 
                     duration=None,
                     trail_length=10,
                     firefly_size=100,
                     blink_color='yellow',
                     normal_color='green',
                     trail_color='lightgreen',
                     background_color='black',
                     save_path=None):
    """
    Create an animation of fireflies moving and blinking, ignoring masked positions.
    
    Args:
        x: Array of x positions, shape (N, T, K) with mask_value indicating invalid positions
        y: Array of y positions, shape (N, T, K) with mask_value indicating invalid positions
        blink: Array of blink states, shape (N, T, K)
        mask_value: Value indicating masked positions (typically 0 or -1)
        fps: Frames per second
        duration: Duration in seconds (if None, uses full length)
        trail_length: Number of previous positions to show in trail
        firefly_size: Size of the firefly markers
        blink_color: Color when firefly is blinking
        normal_color: Color when firefly is not blinking
        trail_color: Color of the trailing positions
        background_color: Color of the plot background
        save_path: If provided, saves animation to this path
    
    Returns:
        matplotlib animation object
    """
    if x.ndim == 2:
        x = x[None, :, :]
        y = y[None, :, :]
        blink = blink[None, :, :]
    N, T, K = x.shape
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=background_color)
    ax.set_facecolor(background_color)
    
    # Set axis limits with some padding, ignoring masked values
    padding = 0.1
    valid_x = x[x != mask_value]
    valid_y = y[y != mask_value]
    x_min, x_max = np.min(valid_x), np.max(valid_x)
    y_min, y_max = np.min(valid_y), np.max(valid_y)
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    
    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Initialize scatter plot for current positions
    scatter = ax.scatter([], [], s=firefly_size, c=normal_color, alpha=0.8)
    
    # Initialize trails (one per firefly)
    trails = []
    for n in range(N):
        for k in range(K):
            trail, = ax.plot([], [], color=trail_color, alpha=0.3, linewidth=1)
            trails.append(trail)
    
    def init():
        scatter.set_offsets(np.c_[[], []])
        for trail in trails:
            trail.set_data([], [])
        return [scatter] + trails
    
    def update(frame):
        # Current positions and blink states
        current_x = x[:, frame, :]
        current_y = y[:, frame, :]
        current_blink = blink[:, frame, :]
        
        # Create mask for valid positions
        valid_mask = current_x != mask_value
        
        # Get valid positions and states
        valid_x = current_x[valid_mask]
        valid_y = current_y[valid_mask]
        valid_blink = current_blink[valid_mask]
        
        # Update scatter plot
        positions = np.c_[valid_x, valid_y]
        colors = np.where(valid_blink, blink_color, normal_color)
        scatter.set_offsets(positions)
        scatter.set_color(colors)
        
        # Update trails
        trail_idx = 0
        for n in range(N):
            for k in range(K):
                # Get trail positions
                start_frame = max(0, frame - trail_length)
                trail_x = x[n, start_frame:frame+1, k]
                trail_y = y[n, start_frame:frame+1, k]
                
                # Only include valid positions in trail
                valid_trail = trail_x != mask_value
                trail_x = trail_x[valid_trail]
                trail_y = trail_y[valid_trail]
                
                trails[trail_idx].set_data(trail_x, trail_y)
                trail_idx += 1
        
        return [scatter] + trails
    
    # Create animation
    frames = T if duration is None else min(T, int(duration * fps))
    anim = FuncAnimation(fig, update, frames=frames, 
                        init_func=init, blit=True, 
                        interval=1000/fps)
    
    # Save animation if path provided
    if save_path:
        anim.save(save_path, fps=fps, writer='pillow')
    
    plt.close()
    return anim


def animate_fireflies_with_images(images, x, y, blink, 
                                mask_value=0,
                                fps=30, 
                                duration=None,
                                trail_length=10,
                                firefly_size=200,
                                blink_color='yellow',
                                normal_color='blue',
                                trail_color='lightgreen',
                                trail_alpha=0.3,
                                save_path=None,
                                use_particle_colors=True):
    """
    Create an animation of fireflies moving and blinking over image frames.
    
    Args:
        images: Array of image frames, shape (N, T, height, width)
        x: Array of x positions, shape (N, T, K) with mask_value indicating invalid positions
        y: Array of y positions, shape (N, T, K) with mask_value indicating invalid positions
        blink: Array of blink states, shape (N, T, K)
        mask_value: Value indicating masked positions (typically 0 or -1)
        fps: Frames per second
        duration: Duration in seconds (if None, uses full length)
        trail_length: Number of previous positions to show in trail
        firefly_size: Size of the firefly markers
        blink_color: Color when firefly is blinking
        normal_color: Color when firefly is not blinking
        trail_color: Color of the trailing positions
        trail_alpha: Alpha value for trails
        save_path: If provided, saves animation to this path
        use_particle_colors: If True, assign a unique color to each particle
        
    Returns:
        matplotlib animation object
    """
    N, T, height, width = images.shape
    _, _, K = x.shape

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Set axis limits to match image dimensions
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates
    
    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Initialize image plot
    img_plot = ax.imshow(np.zeros((height, width)), vmin=0, vmax=1)
    
    # Assign colors for particles
    if use_particle_colors:
        particle_colors = cm.get_cmap('hsv', N)
    else:
        particle_colors = None
    
    # Initialize scatter plots and trails for each particle
    scatters = []
    trails = []
    for n in range(N):
        scatter = ax.scatter([], [], s=firefly_size, alpha=0.8)
        scatters.append(scatter)
        
        particle_trails = []
        for k in range(K):
            trail, = ax.plot([], [], color=trail_color, 
                             alpha=trail_alpha, linewidth=1)
            particle_trails.append(trail)
        trails.append(particle_trails)
    
    def init():
        img_plot.set_array(np.zeros((height, width)))
        for scatter in scatters:
            scatter.set_offsets(np.c_[[], []])
        for particle_trails in trails:
            for trail in particle_trails:
                trail.set_data([], [])
        return [img_plot] + scatters + [trail for particle_trails in trails for trail in particle_trails]
    
    def update(frame):
        # Update image
        img_plot.set_array(images[0, frame])  # Assuming we want to show images for the first sequence
        
        updates = [img_plot]
        for n in range(N):
            # Current positions and blink states for particle n
            current_x = x[n, frame, :]
            current_y = y[n, frame, :]
            current_blink = blink[n, frame, :]
            
            # Create mask for valid positions
            valid_mask = current_x != mask_value
            
            # Get valid positions and states
            valid_x = current_x[valid_mask]
            valid_y = current_y[valid_mask]
            valid_blink = current_blink[valid_mask]
            
            # Update scatter plot
            positions = np.c_[valid_x, valid_y]
            colors = np.where(valid_blink, blink_color, 
                              particle_colors(n) if use_particle_colors else normal_color)
            scatters[n].set_offsets(positions)
            scatters[n].set_color(colors)
            updates.append(scatters[n])
            
            # Update trails
            for k, trail in enumerate(trails[n]):
                start_frame = max(0, frame - trail_length)
                trail_x = x[n, start_frame:frame+1, k]
                trail_y = y[n, start_frame:frame+1, k]
                
                # Only include valid positions in trail
                valid_trail = trail_x != mask_value
                trail_x = trail_x[valid_trail]
                trail_y = trail_y[valid_trail]
                
                trail.set_data(trail_x, trail_y)
                updates.append(trail)
        
        return updates
    
    # Create animation
    frames = T if duration is None else min(T, int(duration * fps))
    anim = FuncAnimation(fig, update, frames=frames, 
                         init_func=init, blit=True, 
                         interval=1000/fps)
    
    # Save animation if path provided
    if save_path:
        anim.save(save_path, fps=fps, writer='pillow')
    
    plt.close()
    return anim


def scatter_animation(observed_xs, observed_ys, gt_xs=None, gt_ys=None, scene_size=32):
    """
    Basic scatter plot animation with moving points
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, scene_size)
    ax.set_ylim(scene_size, 0)
    ax.set_title('Scatter Plot Animation')
    ax.set_facecolor("black")
    
    # Initialize scatter plot
    gt_scatter = ax.scatter([], [], edgecolors='g', facecolors=None, s=200, alpha=0.25, animated=True)
    obs_scatter = ax.scatter([], [], c='red', s=200, animated=True)

    # Animation update function
    def update(frame):
        if gt_xs is not None:
            xs = [x for x in gt_xs[frame, :] if x > 0]
            ys = [y for y in gt_ys[frame, :] if y > 0]
            gt_scatter.set_offsets(np.column_stack([xs, ys]))
            
        xs = [x for x in observed_xs[frame, :] if x > 0]
        ys = [y for y in observed_ys[frame, :] if y > 0]
        # Update scatter plot data
        obs_scatter.set_offsets(np.column_stack([xs, ys]))

        return obs_scatter, gt_scatter
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(observed_xs),  # Number of animation frames
        interval=100,  # Milliseconds between frames
        blit=True
    )
    
    return anim
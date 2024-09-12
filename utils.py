import genjax
from genjax import Mask
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation


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
    observations = list(chm["steps", ..., "observations", "pixels"].value)
    return observations


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
    return ani

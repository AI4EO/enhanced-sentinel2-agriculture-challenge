import gzip
import hashlib
import os
from logging import Logger
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches, patheffects
from shapely.geometry import Polygon, MultiPolygon
from tqdm.auto import tqdm

from eolearn.core import EOPatch

logger = Logger(__file__)


def get_extent(eopatch: EOPatch) -> Tuple[float, float, float, float]:
    """
    Calculate the extent (bounds) of the patch.
    Parameters
    ----------
    eopatch: EOPatch for which the extent is calculated.
    Returns The list of EOPatch bounds (min_x, max_x, min_y, max_y)
    -------
    """
    return eopatch.bbox.min_x, eopatch.bbox.max_x, eopatch.bbox.min_y, eopatch.bbox.max_y


def draw_outline(o, lw, foreground='black'):
    """
    Adds outline to the matplotlib patch.
    Parameters
    ----------
    o:
    lw: Linewidth
    foreground
    Returns
    -------
    """
    o.set_path_effects([patheffects.Stroke(linewidth=lw, foreground=foreground), patheffects.Normal()])


def draw_poly(ax, poly: Union[Polygon, MultiPolygon], color: str = 'r', lw: int = 2, outline: bool = True):
    """
    Draws a polygon or multipolygon onto an axes.
    Parameters
    ----------
    ax: Matplotlib Axes on which to plot on
    poly: Polygon or Multipolygons to plot
    color: Color of the plotted polygon
    lw: Line width of the plot
    outline: Should the polygon be outlined
    Returns None
    -------
    """
    if isinstance(poly, MultiPolygon):
        polys = list(poly)
    else:
        polys = [poly]
    for poly in polys:
        if poly is None:
            logger.warning("One of the polygons is None.")
            break
        if poly.exterior is None:
            logger.warning("One of the polygons has not exterior.")
            break

        x, y = poly.exterior.coords.xy
        xy = np.moveaxis(np.array([x, y]), 0, -1)
        patch = ax.add_patch(patches.Polygon(xy, closed=True, edgecolor=color, fill=False, lw=lw))

    if outline:
        draw_outline(patch, 4)


def draw_bbox(ax, eopatch: EOPatch, color: str = 'r', lw: int = 2, outline: bool = True):
    """
    Plots an EOPatch bounding box onto a matplotlib axes.
    Parameters
    ----------
    ax: Matplotlib axes on which to plot.
    eopatch: EOPatch with BBOx
    color: Color of the BBOX plot.
    lw: Line width.
    outline: Should the plot be additionally outlined.
    Returns None
    -------
    """
    bbox_poly = eopatch.bbox.get_polygon()
    draw_poly(ax, Polygon(bbox_poly), color=color, lw=lw, outline=outline)


def draw_feature(ax, eopatch: EOPatch, time_idx: Union[List[int], int, None], feature: tuple, grid: bool = True,
                 band: int = None, interpolation: str = 'none',
                 vmin: int = 0, vmax: int = 1, alpha: float = 1.0, cmap=plt.cm.viridis):
    """
    Draws an EOPatch feature.
    Parameters
    ----------
    ax: Matplotlib axes on which to plot on
    eopatch: EOPatch for which to plot the mask:
    time_idx: Time index of the mask. If int, single time index of the mask feature, if List[int] multiple masks for
        each time index. If None, plot mask_timeless.
    feature: Tuple defining feature to plot, e.g. (FeatureType.DATA, 'DATA').
    grid: Show grid on plot
    band: Band index of the feature
    interpolation: Interpolation used by imshow
    vmin: Minimum value (for mask visualization)
    vmax: Maximum value (for mask visualization)
    alpha: Transparency of the mask
    cmap: A colormap of the plotted feature
    Returns
    -------
    """

    def _show_single_ts(axis, img, ts):
        fh = axis.imshow(img, extent=get_extent(eopatch), vmin=vmin, vmax=vmax, alpha=alpha, cmap=cmap,
                         interpolation=interpolation)
        if grid:
            axis.grid()
        title = f'{feature[1]} {eopatch.timestamp[ts]}' if ts is not None else f'{feature[1]}'
        axis.set_title(title)
        return fh

    if time_idx is None:
        image = eopatch[feature][..., band] if band is not None else eopatch[feature].squeeze()
        return _show_single_ts(ax, image, time_idx)
    elif isinstance(time_idx, int):
        image = eopatch[feature][time_idx][..., band] if band is not None else eopatch[feature][time_idx].squeeze()
        return _show_single_ts(ax, image, time_idx)
    elif isinstance(time_idx, list):
        for i, tidx in enumerate(time_idx):
            image = eopatch[feature][tidx][..., band] if band is not None else eopatch[feature][tidx].squeeze()
            fh = _show_single_ts(ax[i], image, tidx)
        return fh


def draw_true_color(ax: plt.axes, eopatch: EOPatch, time_idx: Union[List[int], int],
                    feature_name='BANDS-S2-L2A',
                    bands: Tuple[int] = (3, 2, 1),
                    factor: int = 3.5,
                    grid: bool = True):
    """
    Visualization of the bands in the EOPatch.
    Parameters
    ----------
    ax: Axis on which to plot
    eopatch: EOPatch to visualize.
    time_idx: Single timestamp or multiple timestamps.
    feature_name: Name of the feature to visualize.
    bands: Order of the bands.
    factor: Rescaling factor to
    grid: Show grid on visualization
    Returns None
    -------
    """

    def visualize_single_idx(axis, ts):
        axis.imshow(np.clip(eopatch.data[feature_name][ts][..., bands] * factor, 0, 1), extent=get_extent(eopatch))
        if grid:
            axis.grid()
            axis.set_title(f'{feature_name} {eopatch.timestamp[ts]}')

    if isinstance(time_idx, int):
        time_idx = [time_idx]
    if len(time_idx) == 1:
        visualize_single_idx(ax, time_idx[0])
    else:
        for i, tidx in enumerate(time_idx):
            visualize_single_idx(ax[i], tidx)
            

def get_md5(filename: str) -> str:
    """
    Get a MD5 hash code for a specific file.
    Parameters
    ----------
    filename: Path to file
    Returns Hash code
    """
    with gzip.open(filename, 'rb') as bfile:
        md5 = hashlib.md5(np.load(bfile)).hexdigest()
    return md5


def md5_encode_files(path: str) -> pd.DataFrame:
    """
    Encode downloaded numpy files as MD5 hashes.
    Parameters
    ----------
    path: root path to downloaded data
    Returns DataFrame with filenames and associated hashes
    """
    filenames = [os.path.join(root, name) 
                 for root, dirs, files in os.walk(path) for name in files]
    
    filenames = sorted(filenames)
    
    filenames = [filename for filename in filenames if filename.endswith('.npy.gz')]
    
    md5_hashes = [{'filename': filename.replace(f'{path}/', ''),
                   'hash': get_md5(filename)} 
                  for filename in tqdm(filenames)]
    return pd.DataFrame(md5_hashes)

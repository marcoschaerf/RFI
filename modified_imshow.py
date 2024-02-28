def imshow(data, title=None, show=1, cmap=None, norm=None, complex=None, abs=0,
           w=None, h=None, ridge=0, ticks=1, borders=1, aspect='auto', ax=None,
           fig=None, yticks=None, xticks=None, xlabel=None, ylabel=None,save=None, **kw):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    ticks: False to not plot x & y ticks
    borders: False to not display plot borders
    w, h: rescale width & height
    kw: passed to `plt.imshow()`

    others
    """
    # axes
    if (ax or fig) and complex:
        NOTE("`ax` and `fig` ignored if `complex`")
    if complex:
        fig, ax = plt.subplots(1, 2)
    else:
        ax  = ax  or plt.gca()
        fig = fig or plt.gcf()

    # norm
    if norm is None:
        mx = np.max(np.abs(data))
        vmin, vmax = ((-mx, mx) if not abs else
                      (0, mx))
    else:
        vmin, vmax = norm

    # colormap
    import matplotlib as mpl
    mpl33 = bool(float(mpl.__version__[:3]) >= 3.3)
    if cmap is None:
        cmap = (('turbo' if mpl33 else 'jet') if abs else
                'bwr')
    elif cmap == 'turbo':
        if not mpl33:
            from .utils import WARN
            WARN("'turbo' colormap requires matplotlib>=3.3; using 'jet' instead")
            cmap = 'jet'

    _kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect, **kw)

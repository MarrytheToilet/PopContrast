"""The original PopContrast pink/blue publication style, at print-size fonts."""
PINK = '#E06C97'
PINK_SOFT = '#F6C6D8'
PINK_GLOW = '#FBE6EE'
BLUE = '#5B93C9'
BLUE_SOFT = '#C6DCF0'
BLUE_GLOW = '#E7F0FA'
INK = '#39323F'
MUTE = '#8A8494'
BG = '#FDFCFE'
GRID = '#EDE9F1'
GOLD = '#E8B04B'

# Retain Figure 1's palette, with larger type at the actual ACM print size.
COLUMN_WIDTH = 3.335
TEXT_WIDTH = 6.978
REFERENCE_SCALE = COLUMN_WIDTH / (3008 / 300)
REFERENCE_LABEL = 7.5
REFERENCE_TITLE = 8.0
REFERENCE_TICK = 6.5
REFERENCE_LEGEND = 6.5

def apply(plt, size=8):
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
        'axes.edgecolor': '#D9D3E0', 'axes.linewidth': .7,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .65,
        'axes.spines.top': False, 'axes.spines.right': False,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'font.family': 'DejaVu Sans', 'font.size': size,
        'axes.titlesize': size+1, 'axes.titleweight': 'bold',
        'axes.labelsize': size, 'xtick.labelsize': size-.5,
        'ytick.labelsize': size-.5, 'legend.fontsize': size-.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
        'lines.solid_capstyle': 'round',
    })


def apply_reference(plt):
    """Use the original palette with typography specified in printed points."""
    apply(plt, REFERENCE_LABEL)
    plt.rcParams.update({
        'axes.titlesize': REFERENCE_TITLE,
        'axes.labelsize': REFERENCE_LABEL,
        'xtick.labelsize': REFERENCE_TICK,
        'ytick.labelsize': REFERENCE_TICK,
        'legend.fontsize': REFERENCE_LEGEND,
        'axes.linewidth': 1.1 * REFERENCE_SCALE,
        'grid.linewidth': REFERENCE_SCALE,
        'axes.titlepad': 3,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'legend.frameon': False,
        'legend.handlelength': 1.3,
        'legend.handletextpad': .4,
        'legend.columnspacing': .9,
        'hatch.linewidth': .35,
    })

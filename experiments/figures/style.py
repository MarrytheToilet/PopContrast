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

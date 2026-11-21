"""Shared line styles for benchmark evaluation figures."""

TOPAX_COLOR = "#1f77b4"
TOPAX_CONFIGURATION_COLORS = {
    "topax_l4_c2": TOPAX_COLOR,
    "topax_l4_c3": "#d62728",
    "topax_l5_c4": "#9467bd",
}

IMPLEMENTATION_STYLES = {
    "topax": {
        "color": TOPAX_COLOR,
        "marker": "o",
        "linestyle": "-",
        "label": "topax",
    },
    "futtop": {
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "--",
        "label": "Futtop",
    },
    "pyfantom": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "--",
        "label": "pyFANTOM",
    },
    "top3d_xl": {
        "color": "#8c564b",
        "marker": "v",
        "linestyle": "--",
        "label": "TOP3D_XL",
    },
}

TOPAX_CONFIGURATION_STYLES = {
    "topax_l4_c2": {
        "color": TOPAX_CONFIGURATION_COLORS["topax_l4_c2"],
        "marker": "o",
        "linestyle": "-",
        "label": "topax L4-C2",
    },
    "topax_l4_c3": {
        "color": TOPAX_CONFIGURATION_COLORS["topax_l4_c3"],
        "marker": "s",
        "linestyle": "-",
        "label": "topax L4-C3",
    },
    "topax_l5_c4": {
        "color": TOPAX_CONFIGURATION_COLORS["topax_l5_c4"],
        "marker": "D",
        "linestyle": "-",
        "label": "topax L5-C4",
    },
}

WANG_SERIES_STYLES = {
    "topax_l4_c3": {
        "color": "#2ca02c",
        "marker": "s",
        "linestyle": "-",
        "label": "topax L4-C3",
    },
    "topax_l5_c4": {
        "color": "#ff7f0e",
        "marker": "D",
        "linestyle": "-",
        "label": "topax L5-C4",
    },
    "top3d_xl": {
        "color": "#1f77b4",
        "marker": "v",
        "linestyle": "--",
        "label": "TOP3D_XL",
    },
}

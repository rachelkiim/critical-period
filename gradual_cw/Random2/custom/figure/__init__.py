import io
import re
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from pathlib import Path
from lxml import etree


__all__ = ["GraphConfig", "plot_error", "bar_error", "to_px", "to_mm", "load_font"]


class _ConfigMeta(type):
    _initialized = False

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        return cls

    def __getattr__(cls, name):
        if name.startswith("_"):
            return super().__getattribute__(name)

        if not cls._initialized:
            cls._initialize()

        if name in cls.__dict__:
            return cls.__dict__[name]
        else:
            raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

    def __setattr__(cls, name, value):
        super().__setattr__(name, value)

    def _initialize(cls):
        raise NotImplementedError("Subclasses must implement _initialize()")


class GraphConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        cls.color = _ColorConfig


class _ColorConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        cls.DARKGRAY = "#848484"
        cls.LIGHTGRAY = "#F7F7F7"
        cls.ORANGE = "#DF6347"
        cls.SKY = "#0989FF"
        cls.BLUE = "#004C98"
        cls.SAND = "#B29166"
        cls.GREEN = "#495F39"
        cls.PINK = "#B26666"
        cls.BROWN = "#634848"
        cls.PURPLE = "#514864"
        cls.RUST = "#790000"
        cls.OCEAN = "#003049"
        cls.CRIMSON = "#C1111F"
        cls._initialized = True


def plot_error(data, color, label=None, xrange=None):
    """
    Plot the mean and standard deviation of the data.

    This function creates a line plot of the mean values with a shaded region
    representing the standard deviation around the mean.

    Args:
        data: The input data array where axis=0 represents separate measurements.
        color: The color to use for the mean line and shaded error region.
        label: The label for the plot legend. If None, no label is assigned.
        xrange: The x-coordinates for the data points. If None, indices starting

    Returns:
        The transparent PolygonCollection representing the shaded error region.
    """

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    upper, lower = mean + std, mean - std

    if xrange is None:
        xrange = np.arange(len(mean))

    if label is None:
        plt.plot(xrange, mean, color=color)
    else:
        plt.plot(xrange, mean, color=color, label=label)

    return plt.fill_between(
        xrange, upper, lower, color=color, alpha=0.2, edgecolor=None
    )


def bar_error(xticks, data, color):
    """
    Create a bar plot with error bars showing mean and standard deviation.

    Parameters:
    -----------
    xticks : array-like
        The x positions of the bars.
    data : array-like
        The data to plot, where axis=0 represents separate measurements.
    color : str or tuple
        The color of the bars.

    Returns:
    --------
    matplotlib.container.BarContainer
        The bar container object with all bars and error bars.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return plt.bar(
        xticks, mean, yerr=std, color=color, width=0.5, error_kw={"linewidth": 0.5}
    )


_sup_map = {
    '0': '⁰','1': '¹','2': '²','3': '³','4': '⁴','5': '⁵','6': '⁶','7': '⁷','8': '⁸','9': '⁹',
    '+': '⁺','-': '⁻','=': '⁼','(': '⁽',')': '⁾','n': 'ⁿ'
}
_sub_map = {
    '0': '₀','1': '₁','2': '₂','3': '₃','4': '₄','5': '₅','6': '₆','7': '₇','8': '₈','9': '₉',
    '+': '₊','-': '₋','=': '₌','(': '₍',')': '₎'
}


def _convert_mathdefault(text: str) -> str:
    def repl_math(m):
        inner = m.group(1)
        inner = re.sub(r"\\times", "×", inner)

        inner = re.sub(
            r"\^\{(-?\d+)\}",
            lambda mm: ''.join(_sup_map.get(c, c) for c in mm.group(1)),
            inner
        )
        inner = re.sub(
            r"_\{(-?\d+)\}",
            lambda mm: ''.join(_sub_map.get(c, c) for c in mm.group(1)),
            inner
        )
        return inner

    text = re.sub(r"\$\\mathdefault\{(.+?)\}\$",  repl_math, text)
    text = re.sub(r"\\mathdefault\{(.+?)\}",       repl_math, text)

    text = re.sub(r"\\times", "×", text)

    return text


def _convert_svg_text(svg_content: str) -> str:
    parser = etree.XMLParser(remove_comments=False)
    tree = etree.parse(io.BytesIO(svg_content.encode("utf-8")), parser)
    root = tree.getroot()
    namespaces = root.nsmap
    svg_ns = namespaces.get(None)
    nsmap = {"svg": svg_ns} if svg_ns else {}

    figure_group = root.xpath("//svg:g[@id='figure_1']", namespaces=nsmap)
    if figure_group:
        fg = figure_group[0]
        for child in list(fg):
            fg.remove(child)
            root.append(child)
        parent = fg.getparent()
        if parent is not None:
            parent.remove(fg)

    for text_group in root.xpath("//svg:g[starts-with(@id, 'text')]", namespaces=nsmap):
        raw_txt = text_group.xpath("comment()")[0].text.strip()
        raw_txt = _convert_mathdefault(raw_txt)
        raw_txt = raw_txt.strip("$")

        transform_groups = text_group.xpath(
            ".//svg:g[@transform]" if svg_ns else ".//g[@transform]",
            namespaces=nsmap
        )
        transform_attr = transform_groups[0].get("transform", "")
        translate_match = re.search(r"translate\(([-\d.]+)\s+([-\d.]+)\)", transform_attr)
        rotate_match    = re.search(r"rotate\(([-\d.]+)\)", transform_attr)
        scale_match     = re.search(r"scale\(([\d.]+)\s+-([\d.]+)\)", transform_attr)

        x_pos = translate_match.group(1)
        y_pos = translate_match.group(2)
        rotate = rotate_match.group(1) if rotate_match else None
        scale_x = float(scale_match.group(1)) if scale_match else 1.0
        scale_y = float(scale_match.group(2)) if scale_match else 1.0
        font_size = int(max(scale_x, scale_y) * 100)

        for child in list(text_group):
            text_group.remove(child)

        text_group.append(etree.Comment(raw_txt))

        tag = "text" if not svg_ns else f"{{{svg_ns}}}text"
        text_elem = etree.SubElement(text_group, tag)
        text_elem.set("x", x_pos)
        text_elem.set("y", y_pos)
        text_elem.set("font-family", "Arial")
        text_elem.set("font-size", f"{font_size}px")
        text_elem.set("font-weight", "regular")
        text_elem.set("fill", "currentColor")
        text_elem.set("text-anchor", "start")

        text_elem.set("dominant-baseline", "alphabetic")
        if rotate:
            text_elem.set("transform", f"rotate({rotate}, {x_pos}, {y_pos})")

        text_elem.text = raw_txt

    converted_content = etree.tostring(
        root, encoding="utf-8", pretty_print=True
    ).decode("utf-8")
    converted_content = converted_content.replace("pt", "px")

    return converted_content


def to_px(value: int) -> float:
    """
    convert px to matplotlib inch
    """
    return value * 0.01803751804


def to_mm(value: int) -> float:
    """
    convert mm to matplotlib inch
    """
    return value * 0.05110624098


def load_font(font_path: str):
    """
    Load a font file to matplotlib.

    Args:
        font_path: The path to the font file.
    """
    font_manager.fontManager.addfont(font_path)


def savefig(input_file: str):
    """Override of plt.savefig that converts vector-based text to actual text elements in SVG files.

    This function extends the original matplotlib savefig functionality by post-processing
    SVG files to improve text rendering. It converts vector-based text elements to actual
    text elements and changes the SVG bounding box metric from 'pt' to 'px' for better
    rendering consistency.

    Args:
        input_file: Path to the file where the figure should be saved. If the file
            is an SVG file, text conversion will be applied after saving.

    Note:
        If the input file is not an SVG file, a warning message is logged and
        no conversion is performed.
    """
    input_path = Path(input_file)
    _original_savefig(input_file)

    if input_path.suffix != ".svg":
        logging.warning(f"Input file '{input_file}' is not an svg file. skipping...")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        svg_content = f.read()

    converted_content = _convert_svg_text(svg_content)

    with open(input_path, "w", encoding="utf-8") as f:
        f.write(converted_content)

def _set_format(self):
    self.format = '%g'
matplotlib.ticker.ScalarFormatter._set_format = _set_format


# override plt.savefig to save svg file with converted text
_original_savefig = plt.savefig
plt.savefig = savefig

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.style"] = "normal"
plt.rcParams["font.size"] = 6

# axis
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.facecolor"] = (0, 0, 0, 0)
plt.rcParams["axes.autolimit_mode"] = "round_numbers"
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0
plt.rcParams["axes.linewidth"] = 0.75
plt.rcParams["axes.formatter.use_mathtext"] = False

# savefig
plt.rcParams["savefig.facecolor"] = (0, 0, 0, 0)
plt.rcParams["savefig.transparent"] = True

# figure
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.facecolor"] = (0, 0, 0, 0)

# ytick
plt.rcParams["ytick.labelsize"] = 6
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["ytick.major.size"] = 2
plt.rcParams["ytick.major.width"] = 0.75
plt.rcParams["ytick.major.pad"] = 3

# xtick
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["xtick.major.size"] = 2
plt.rcParams["xtick.major.width"] = 0.75
plt.rcParams["xtick.major.pad"] = 3

# line
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["lines.color"] = "black"

# scatter
plt.rcParams["scatter.marker"] = "."

# boxplot
plt.rcParams["boxplot.boxprops.linewidth"] = 0
plt.rcParams["boxplot.whiskerprops.linewidth"] = 0.5
plt.rcParams["boxplot.whiskerprops.linestyle"] = "-"
plt.rcParams["boxplot.whiskerprops.color"] = "black"
plt.rcParams["boxplot.medianprops.linewidth"] = 0.5
plt.rcParams["boxplot.medianprops.color"] = "white"
plt.rcParams["boxplot.capprops.linewidth"] = 0.5
plt.rcParams["boxplot.flierprops.marker"] = "o"
plt.rcParams["boxplot.flierprops.markerfacecolor"] = "darkgray"
plt.rcParams["boxplot.flierprops.markersize"] = 3
plt.rcParams["boxplot.flierprops.linestyle"] = "none"
plt.rcParams["boxplot.flierprops.markeredgecolor"] = "none"
plt.rcParams["boxplot.patchartist"] = True
plt.rcParams["boxplot.showfliers"] = False
plt.rcParams["boxplot.notch"] = False

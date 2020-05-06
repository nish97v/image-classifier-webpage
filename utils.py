from numpy import pi
from bokeh.plotting import figure
from bokeh.embed import components


def is_allowed_file(filename, allowed_ext):
    """
    Checks if a filename's extension is acceptable
    :param filename: str - name of the uploaded file
    :param allowed_ext: set - allowed extensions
    :return: bool
    """
    ext = filename.rsplit('.', 1)[1].lower() in allowed_ext
    return ext and '.' in filename


def generate_barplot(predictions, labels):
    """
    Generates script and 'div' element of bar plot of predictions using Bokeh
    :param predictions: prediction probability values for each class
    :return: tuple of generated JavaScript and the <div> elements for the plot
    """
    plot = figure(x_range=labels, plot_height=300, plot_width=400)
    plot.vbar(x=labels, top=predictions, width=0.8)
    # plot.xaxis.major_label_orientation = pi / 2.
    # plot.xaxis.axis_label_text_font_size = "40pt"
    # plot.yaxis.axis_label_text_font_size = "40pt"

    return components(plot)
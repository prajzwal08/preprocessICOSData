import matplotlib.pyplot as plt

def generate_line_plot(x, y, label='', xaxislabel='', yaxislabel='', title='', output_path=''):
    """
    Generate and save a line plot.

    Parameters:
        x (array-like): X-axis data points.
        y1 (array-like): Y-axis data points for the first line plot.
        y2 (array-like, optional): Y-axis data points for the second line plot.
        label1 (str): Label for the first line plot.
        label2 (str): Label for the second line plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title for the plot.
        output_path (str): Path to save the output plot.
    """
    plt.clf()
    plt.rcParams.update({
        'font.size': 12,
    })
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y,
             linestyle='--',
             linewidth=1.8,
             color='red',
             label=label)
        
    plt.xlabel(xaxislabel)
    plt.ylabel(yaxislabel)
    plt.title(title)
    plt.grid(linestyle='dotted')
    plt.legend(ncol=2, loc='upper right')  # 9 means top center
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()
    
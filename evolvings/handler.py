

class Handler(object):

    def __init__(self):
        pass

    def insert_legend(self, ax=None, loc='upper left',
                      fontsize='large', linewidth=1.5):
        legend = ax.legend(loc=loc, shadow=True)

        frame = legend.get_frame()
        frame.set_facecolor('0.90')

        for label in legend.get_texts():
            label.set_fontsize(fontsize)

        for label in legend.get_lines():
            label.set_linewidth(linewidth)

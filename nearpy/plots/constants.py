from lets_plot import *

DEFAULT_PLOT_STYLE = theme(axis_title_x=element_text(size=18, face='bold', family='sans'),
                    axis_text_x=element_text(size=16, family='sans'),
                    axis_title_y=element_text(size=18, face='bold', family='sans'),
                    axis_text_y=element_text(size=16, family='sans'),
                    axis_ticks_length=10,
                    plot_title=element_text(size=24, face='bold', family='sans', hjust=0.5),
                    panel_border=element_rect(color='black', size=2),
                    plot_margin=10,
                    panel_background=element_rect(color='black', fill='#eeeeee', size=2),
                    panel_grid=element_line(color='black', size=1))

def BBC_THEME(show_x_axis=True):
    line_size = 1

    def get_element_text(title=False, subtitle=False, size=21):
        face = None
        text_margin = None
        if title:
            size = 24
            face = "bold"
            text_margin = [11, 0, 0, 0]
        if subtitle:
            size = 20
            text_margin = [9, 0, 0, 0]
        return element_text(family="Helvetica", face=face, size=size, margin=text_margin)
    result = theme(
        plot_title=get_element_text(title=True),
        plot_subtitle=get_element_text(subtitle=True),
        legend_position='top',
        legend_background='blank',
        legend_title='blank',
        legend_text=get_element_text(),
        axis_title='blank',
        axis_text=get_element_text(),
        axis_text_x=element_text(margin=[20, 20]),
        axis_text_y=element_text(margin=[10, 5]),
        axis_ticks='blank',
        axis_line=element_line(size=2*line_size) if show_x_axis else 'blank',
        axis_ontop_x=True,
        panel_grid_minor='blank',
        panel_grid_major_y=element_line(size=line_size, color='#CBCBCB'),
        panel_grid_major_x='blank',
        panel_background='blank',
        strip_text=element_text(size=16, hjust=0),
    )
    # if show_x_axis:
    #     result += coord_cartesian(ylim=[0, None]) + scale_y_continuous(expand=[.15, 0])

    return result
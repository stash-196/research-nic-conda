import plotly.express as px


def visualize_output(
    df, p="target_p", y="target_y", fig_show=False, save_to_directory_as=None
):
    fig_p = px.scatter(df, x="x1", y="x2", color=p, range_color=[0, 1])
    fig_y = px.scatter(df, x="x1", y="x2", color=y, opacity=0.7)
    if fig_show:
        fig_p.show()
        fig_y.show()
    if save_to_directory_as is not None:
        fig_p.write_image(save_to_directory_as + "_p.png")
        fig_y.write_image(save_to_directory_as + "_y.png")

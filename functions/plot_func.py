from matplotlib.colors import LinearSegmentedColormap

def my_ridgeline(x_list,data_list,ax,label,linewidth,ouline_color='k',gradient_colors = ['indigo', 'lightblue']):
    eps = (max(data_list[0]) - min(data_list[0])) / 100.
    n_data = len(data_list)
    offset_constant =   4*(max(data_list[0]) - min(data_list[0])) / n_data

    cmap_name = 'my_gradient'
    n_bins = 100    
    cmap = LinearSegmentedColormap.from_list(cmap_name, gradient_colors,N=n_bins)

    for n,data in enumerate(data_list):
        minim = min(data)
        maxim = max(data)
        offset_data = [y + n * offset_constant for y in data]
        offset = [minim-eps + n * offset_constant for _ in x_list]        
        ax.fill_between(x_list,offset,offset_data, zorder=n_data-n,color=cmap(float(n/(n_data-1))), alpha=1)
        if n == 0:
            ax.plot(x_list, offset_data, zorder=n_data-n,color = ouline_color, alpha=1,linewidth=linewidth,label=label)
        else:
            ax.plot(x_list, offset_data, zorder=n_data-n,color = ouline_color, alpha=1,linewidth=linewidth)
    return min(data_list[0]) - eps, 2*max(data_list[-1]) + n_data * offset_constant, eps, offset_constant
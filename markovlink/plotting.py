def Ms_plot(Ms,x,additional_rows=None,additional_labels=None):
    import matplotlib.pylab as plt

    idx=5

    plt.gcf().add_axes([0,.6,1,1])
    plt.pcolor(Ms[:,x],vmin=0,vmax=1); plt.yticks([]); plt.xticks([])
    plt.ylabel("$q$s from the confidence interval")
    plt.title("Inspecting $q(\\cdot | %d)$"%idx)

    plt.gcf().add_axes([0,.3,1.0,.2])
    plt.pcolor(qtilde[idx,None],vmin=0,vmax=1); plt.yticks([]); plt.xticks([])
    plt.ylabel("$\\tilde q$")

    plt.gcf().add_axes([0,0,1.0,.2])
    mappable=plt.pcolor(qxy_star[idx,None],vmin=0,vmax=1); plt.yticks([]); plt.xticks([])
    plt.ylabel("$q^*$")

    plt.gcf().add_axes([1.1,0,.1,1.6])
    plt.colorbar(mappable,cax=plt.gca())
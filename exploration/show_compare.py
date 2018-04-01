
def visualize_dataset(path1, path2, plot):
    data = np.genfromtxt(dataset_path, delimiter=',')
    ppg_signal1 = data[:,1].flatten()
    breath_signal = data[:,3].flatten()
    ppg_signal2 = data[:,1].flatten()
    print(ppg_signal.shape)

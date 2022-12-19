import matplotlib.pyplot as plt

col_names = ['Variance_WT', 'Skewness_WT', 'Curtosis_WT', 'Entropy', 'Class']

def plotting_data(data_authentic=[], data_inauthentic=[]):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
  X_plot = [t[0] for t in data_authentic]
  Y_plot = [t[1] for t in data_authentic]

  ax1.plot(X_plot, Y_plot, 
            linestyle='none', 
            marker='o', 
            label="Authentic")
  X_plot = [t[0] for t in data_inauthentic]
  Y_plot = [t[1] for t in data_inauthentic]
  ax1.plot(X_plot, Y_plot, 
            linestyle='none', 
            marker='o', 
            label="Inauthentic")
  ax1.set_xlabel('Variance_WT')
  ax1.set_ylabel('Skewness_WT')
  ax1.axis('equal')
  ax1.legend();

  X_plot = [t[2] for t in data_authentic]
  Y_plot = [t[3] for t in data_authentic]
  ax2.plot(X_plot, Y_plot, 
            linestyle='none', 
            marker='o', 
            label="Authentic")
  X_plot = [t[2] for t in data_inauthentic]
  Y_plot = [t[3] for t in data_inauthentic]
  ax2.plot(X_plot, Y_plot, 
            linestyle='none', 
            marker='o', 
            label="Inauthentic")
  ax2.set_xlabel(col_names[2])
  ax2.set_ylabel(col_names[3])
  ax2.axis('equal')
  ax2.legend();
  plt.show()
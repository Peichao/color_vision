import numpy as np
import matplotlib.pyplot as plt
import functions

animal = 'AE4'
penetration = '004'
exp_ach = '005'
exp_L = '002'
exp_M = '003'
exp_S = '004'
data_folder = 'H:/AE4/004'
cluster = '964'
plot_ind = 160

dir_ach = data_folder + '/%s_u%s_%s/' % (animal, penetration, exp_ach)
dir_L = data_folder + '/%s_u%s_%s/' % (animal, penetration, exp_L)
dir_M = data_folder + '/%s_u%s_%s/' % (animal, penetration, exp_M)
dir_S = data_folder + '/%s_u%s_%s/' % (animal, penetration, exp_S)

revcorr_ach = np.load(dir_ach + 'revcorr_images_%s.npy' % cluster)
revcorr_L = np.load(dir_L + 'revcorr_images_%s.npy' % cluster)
revcorr_M = np.load(dir_M + 'revcorr_images_%s.npy' % cluster)
revcorr_S = np.load(dir_S + 'revcorr_images_%s.npy' % cluster)

plot_lim_ach = functions.plot_lim(revcorr_ach.min(), revcorr_ach.max())
plot_lim_L = functions.plot_lim(revcorr_L.min(), revcorr_L.max())
plot_lim_M = functions.plot_lim(revcorr_M.min(), revcorr_M.max())
plot_lim_S = functions.plot_lim(revcorr_S.min(), revcorr_S.max())
plot_lim = np.array([plot_lim_ach, plot_lim_L, plot_lim_M, plot_lim_S]).max()

fig, (ax_ach, ax_L, ax_M, ax_S) = plt.subplots(1, 4)

ax_ach.imshow(revcorr_ach[:, :, plot_ind], cmap='jet', interpolation='bilinear', vmin=-plot_lim, vmax=plot_lim)
ax_ach.axis('off')
ax_ach.set_title('Achromatic')
ax_L.imshow(revcorr_L[:, :, plot_ind], cmap='jet', interpolation='bilinear', vmin=-plot_lim, vmax=plot_lim)
ax_L.axis('off')
ax_L.set_title('L Cone')
ax_M.imshow(revcorr_M[:, :, plot_ind], cmap='jet', interpolation='bilinear', vmin=-plot_lim, vmax=plot_lim)
ax_M.axis('off')
ax_M.set_title('M Cone')
ax_S.imshow(revcorr_S[:, :, plot_ind], cmap='jet', interpolation='bilinear', vmin=-plot_lim, vmax=plot_lim)
ax_S.axis('off')
ax_S.set_title('S Cone')

plt.savefig(data_folder + '/hartley_cluster%s.pdf' % cluster, format='pdf')

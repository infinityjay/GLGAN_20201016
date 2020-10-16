import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

c = np.load('c_data.npy') #size c = 17101
x1 = np.linspace(0,100,17101)
loss1 = plt.plot(x1,c, c='c', ls=':',lw=0.5, label='c_loss' )
plt.ylim(0,6000)
plt.xlabel('Epoch',fontsize = 12)
plt.ylabel('c_loss', fontsize = 12)
plt.grid(b = True, color = 'k', linestyle = '--', alpha = 0.2)#alpha 为透明度
plt.legend(fontsize = 10)
ax = plt.gca()
ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.savefig('./loss_plot/tc100_td1_trainstep400/c_loss.tif')
plt.show()


# #画d_loss

# d = np.load('d_data.npy') #size d = 171
# x2 = np.linspace(0,171,171)
# plt.plot(x2,d, c='b', ls=':',lw=0.5, label='d_loss')
# plt.legend()
# plt.show()

#画g_loss
g = np.load('g_data.npy') #size g = 51129
x3 = np.linspace(100,400,51129)
plt.plot(x3,g, c='orange', ls=':',lw=0.5, label='g_loss')
plt.xlabel('Epoch',fontsize = 12)
plt.ylabel('g_loss', fontsize = 12)
plt.grid(b = True, color = 'w', linestyle = '-', alpha = 1)#alpha 为透明度

ax = plt.gca()
ax.patch.set_facecolor("gray") #设置背景颜色
ax.patch.set_alpha(0.2)
plt.legend(fontsize = 10)
ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.savefig('./loss_plot/tc100_td1_trainstep400/g_loss.tif')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# # 求closs在每个epoch中的平均值
# c_ave = []
# c = np.load('./loss_plot/tc300_td1_trainstep1000/c_data.npy') #size c = 17101
# for i in range(300):
#     sum = 0
#     for j in range(171):
#         sum = sum + c[171*i+j]
#     ave = sum / 171
#     c_ave.append(ave)
# # print(c_ave)
# # np.save('./loss_plot/tc300_td1_trainstep1000/c_ave.npy', c_ave)
# # 画closs的图
# x1 = np.linspace(0,300,300)
# loss1 = plt.plot(x1,c_ave, c='c', marker='o', ms= 2, lw=0.5, label='c_loss' )
# plt.ylim(0,6000)
# plt.xlabel('Epoch',fontsize = 12)
# plt.ylabel('c_loss', fontsize = 12)
# plt.grid(b = True, color = 'k', linestyle = '--', alpha = 0.2)#alpha 为透明度
# plt.legend(fontsize = 10)
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# # ax.spines['bottom'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # ax.spines['left'].set_visible(False)
# # plt.savefig('./loss_plot/tc100_td1_trainstep400/c_loss_ave.tif')
# # plt.show()


# 求gloss在每个epoch中的平均值
g_ave = []
g1 = np.load('./loss_plot/tc300_td1_trainstep1000/g_data.npy')
g2 = np.load('./loss_plot/tc300_td1_trainstep1000/g_data_2.npy')
g2 =np.append(g1,g2)
g3 = np.load('./loss_plot/tc300_td1_trainstep1000/g_data_3.npy')
g =np.append(g2,g3)
for i in range(2699):
    sum = 0
    for j in range(171):
        sum = sum + g[171*i+j]
    ave = sum / 171
    g_ave.append(ave)
# print(g_ave)
# np.save('./loss_plot/tc300_td1_trainstep1000/g_ave.npy', g_ave)
# 画gloss的图
x2 = np.linspace(301,3000,2699)
loss1 = plt.plot(x2,g_ave, c='orange', marker='x', ms= 0.2, lw=0.2, label='g_loss' )
# plt.ylim(0,6000)
plt.xlabel('Epoch',fontsize = 12)
plt.ylabel('g_loss', fontsize = 12)
plt.grid(b = True, color = 'w', linestyle = '--', alpha = 0.2)#alpha 为透明度
plt.legend(fontsize = 10)
ax = plt.gca()
ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.savefig('./loss_plot/tc300_td1_trainstep1000/g_loss_ave.tif')
plt.show()
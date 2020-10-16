1.数据集说明
1.1 训练过程
data中包含有train文件（训练数据集）test文件（测试数据集）
1.2 测试评估
1.2.1 mask过程测试评估数据集
共有2014、2016、2018三个年份，每个年份有6个文件夹，下面以2014年为例进行说明：
data_2014 为原始大小2014年1月1日到3月31日1080张图片
2014_orig 为64·64大小2014年1月1日到3月31日1080张图片
2014_test 为64·64大小mask操作之后的图片
2014_result 为64·64大小用当前checkpoint中模型跑出来的补全后图片
2014_cut_orig为mask操作时剪切下来的图片
2014_cut_res 为补全后的图片在mask操作时的同样的位置剪切的图片
1.2.2 手动mask测试评估集
manual_orig为原始数据集
manual_test为手动mask之后的数据集
manual_res为补全之后的数据集
1.3 结果
loss_plot文件夹中包含有不同训练参数和不同训练次数对应的三个loss，其中c_loss为补全网络损失,d_loss为判别器损失（基本不画数据量小）,g_loss为全局损失

2.固定下来的模型
checkpoints为当前保存的模型是训练4000后较为稳定的模型可以直接调用对图片进行补全
other_checkpoints为其他训练次数保存的模型

3.程序文件功能
architecture.py 定义了构建网络中用到的几个函数
config.py 定义了一些参数的格式、具体数值以及解释说明
cut_test.py 为批量mask后补全图片的程序并且可以保存mask过程中裁剪出来图片以及补全后同样位置裁剪出来的图片
download.py 下载模型的程序，但是没有用到
network.py 为网络构建程序，在进行train和test的时候因为placeholder的原因，需要修改
ops.py 为传入network.py中的几个读入数据的函数
train.py为训练程序
test.py为手动mask时的测试程序
plot.py、plot2.py都为绘制不同loss图的程序

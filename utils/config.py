import math
Config = {
    #-----------------------------------------------------------------#
    #   输入尺寸为480*640
    #   输入的图像尺寸为长方形，feature_maps也要转换为长方形
    #   各个参数的理解参考：https://blog.csdn.net/xunan003/article/details/79186162
    #   steps：即计算卷积层产生的prior_box距离原图的步长，先验框中心点的坐标会乘以step，相当于从feature map位置映射回原图位置，
    #          比如conv4_3输出特征图大小为38*38，而输入的图片为300*300，所以38*8约等于300，所以映射步长为8。这是针对300*300的训练图片
    #   min_sizes：检测框大小（min）,SSD-300:[30, 60, 111, 162, 213, 264]
    #   max_sizes：检测框大小（max）,SSD-300:[60, 111, 162, 213, 264, 315]
    #   min_sizes,max_sizes计算方式见主程序
    #   aspect_ratios：不同特征图上检测框绘制比例，横纵比，确定每个中心点产生多少个检测框，
    #   计算方式如下：在SSD中6层卷积层的每个特征图的每个中心点会产生2个不同大小的正方形默认框，另外每设置一个aspect_ratio则会增加两个长方形默认框
    #                 第一层默认生成两个不同大小的正方形默认框，另外又有一个aspect_ratio=2产生了两个长方形默认框，所以总共有4个。
    #                 再如第二层，默认生成两个正方形默认框，另外又有aspect_ratio=[2,3]，所以又生成了4个不同的长方形默认框，共有6个不同大小的默认框
    #   variance：center_variance, size_variance 解码
    #   clip：True ,检测框越界截断.  0<检测框尺寸<480

    #-----------------------------------------------------------------#
    'min_dim': 480,
    # 'feature_maps': [38, 19, 10, 5, 3, 1],
    # 'min_dim': 512,
    # 'feature_maps': [64, 32, 16, 8, 6, 4],
    'feature_maps': [(60, 80), (30, 40), (15, 20), (7, 9), (3, 4), (1, 1)],
    'steps': [(8, 8), (16, 16), (32, 32), (68, 71), (160, 160), (480, 640)],
    'min_sizes': [48, 96, 178, 259, 341, 422],
    'max_sizes': [96, 178, 259, 341, 422, 504],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

if __name__ =='__main__':
    min_ratio = 20  ####这里即是论文中所说的Smin=0.2，Smax=0.9的初始值，经过下面的运算即可得到min_sizes，max_sizes。具体如何计算以及两者代表什么，请关注我的博客SSD详解。这里产生很多改进。
    max_ratio = 90
    step = int(math.floor((max_ratio - min_ratio) / (6 - 2)))

    min_sizes = []  ###经过以下运算得到min_sizes和max_sizes。
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(300 * ratio / 100.)
        max_sizes.append(300 * (ratio + step) / 100.)

    min_sizes = [300 * 10 / 100.] + min_sizes
    max_sizes = [300 * 20 / 100.] + max_sizes
    print('minsizes:', min_sizes)
    # minsizes: [48.0, 96.0, 177.6, 259.2, 340.8, 422.4]
    print('maxsizes:', max_sizes)
    # maxsizes: [96.0, 177.6, 259.2, 340.8, 422.4, 504.0]

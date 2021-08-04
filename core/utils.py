# -*- coding: utf-8 -*-
# @File : utils.py
# @Author: Runist
# @Time : 2021/7/16 18:07
# @Software: PyCharm
# @Brief:


def process_bar(block_num, block_size, total_size):
    """
    :param block_num: 已经下载的数据块
    :param block_size: 数据块的大小
    :param total_size: 远程文件的大小
    """

    percent = block_num / (total_size / block_size)

    a = "=" * int(percent * 30)
    b = "." * int((1 - percent) * 30)

    if percent >= 1.0:
        print("\r100%[{}]\n".format("="*30))
    else:
        print("\r{:^3.0f}%[{}>{}]".format(int(percent * 100), a, b), end="")

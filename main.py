# -*- coding:utf-8 -*-
# @Author:wch
# @time:2022/11/21

import chardet
import struct
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy.fftpack import fft
import numpy as np
import pywt
import pandas as pd
import math
from math import log

#获得文件的编码
def get_encoding(file):
    with open(file,'rb') as f:
        return chardet.detect(f.read(1))['encoding']

# 获取以.D0结尾的文件的波形，用列表保存，大小为360000
def get_data(path):
    # 输入文件路径，
    # curveData为存储了[3600s * 100hz]的列表，代表电压值大小，mesage是文件头部的信息，打印会自动使用ASCII编码转为为文本信息
    f = open(path, "rb")  # 以二进制方式读取文件
    rawData = f.read()
    f.close()
    iSampleCount = 360000
    # print("文件信息+采集点数：", len(rawData))
    MesageLen = (len(rawData) - 360000 * 4)
    # print(MesageLen)
    curveData = []
    for i in range(iSampleCount):
        fValue, = struct.unpack("<f", rawData[MesageLen + i * 4:MesageLen + 4 + i * 4])
        curveData.append(fValue)
    mesage = rawData[:MesageLen]
    return curveData,mesage

# 如果有需要，可用把该函数的10s限制关闭
def fft_Trans(rawData,begin=0):
    # 对10s的波形进行离散傅里叶变换，得到波形的频谱图
    if begin < 350000:
        data_freq = fft(rawData[begin*100:begin*100+1000])
    else:
        data_freq = fft(rawData[begin*100:360000])
    mdata = np.abs(data_freq)  # magnitude  取模
    pdata = np.angle(data_freq)  # phase
    # plt.subplot(2, 2, 3)
    # plt.plot(mdata)
    #
    # # plt.subplot(2, 2, 4)
    # # plt.plot(pdata)
    # plt.show()
    return mdata,pdata

# 如果有需要，可用把该函数的10s限制关闭
def wavelet_Trans(rawData, begin=0):
    # 对10s的时域信号进行（连续）小波变换，得到时频谱
    """
    :param curveData: EPG data
    :param begin: 开始的时间
    :return:
    连续小波变换中可用的小波：
    **A wide range of continous wavelets are now available. These include the following:**
	Gaussian wavelets (gaus1…``gaus8``)
	Mexican hat wavelet (mexh)
	Morlet wavelet (morl)
	Complex Gaussian wavelets (cgau1…``cgau8``)
	Shannon wavelet (shan)
	Frequency B-Spline wavelet (fbsp)
	Complex Morlet wavelet (cmor)
	---------------------------------------------
	              还没做输入的鲁棒性
	---------------------------------------------
    """

    sampling_rate = 100
    t = np.arange(0, 10, 1.0 / sampling_rate)  # star：0  end：1   f = 1/sampling   等分
    data = rawData[begin*100:begin*100+1000]  # 600*100

    wavename = 'morl'
    totalscal = 256
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
    plt.figure(figsize=(8, 4))
    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.ylabel(u"prinv(Hz)")
    plt.xlabel(u"t(s)")
    plt.subplots_adjust(hspace=0.4)
    plt.show()

# 没啥用的小波阈值降噪
def DWT_Threshold(curveData,begin=0):

    if begin < 350000:
        data = curveData[begin*100:begin*100+1000]
    else:
        data = fft(curveData[begin*100:360000])
    # plt.subplot(2, 1, 1)
    # plt.plot(data)
    w = pywt.Wavelet('coif4')  # 选用小波
    # print(w)
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    # print("maximum level is " + str(maxlev))
    threshold = 0.4  # Threshold for filtering
    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'coif4', level=maxlev)  # 将信号进行小波分解
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波
    datarec = pywt.waverec(coeffs, 'coif4')
    # plt.subplot(2, 1, 2)
    # plt.plot(datarec)
    # plt.show()

    return str(maxlev),datarec

def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

def rigrsure(data,N):
    """
    :param data: 波
    :param N: len(data)
    :return: 返回未乘噪声方差的阈值
    """
    if type(data) == list:
        data = np.array(data)
    N=len(data)
    # 风险阈值
    sx = np.sort(abs(data))
    sx2 = np.square(sx)  # P37  NV
    N1 = np.repeat(N - 2 * np.array([i for i in range(0, N)]), 1)  # N-2i
    N2 = np.repeat(N - np.array([i for i in range(0, N)]), 1)  # N-i
    CS1 = np.cumsum(sx2, 0)  # f(i)求和
    risks = (N1 + CS1 + np.multiply(N2, sx2)) / N
    # 过滤小于0
    for i in range(len(risks)):
        if risks[i] < 0:
            risks[i] = 0.3
    min = np.argmin(risks,0)
    # thr will be row vector
    thr = np.sqrt(sx2[min])
    # print(lamda)
    return thr

def thrselect(data,thr_select):
    """
    阈值lambda的计算方式选择
    :return: 返回未乘噪声方差的阈值
    """
    if thr_select in ['rigrsure', 'sqtwolog', 'minimaxi']:
        thr_select = thr_select
    else:
        raise print('取值计算函数名称错误，请重新输入')

    N = 360000
    if thr_select == 'sqtwolog':
        # 固定阈值
        thr = np.sqrt(2.0 * np.log(N))
        # print("固定阈值=",thr)
        return thr

    elif thr_select == 'minimaxi':
        # 极大极小阈值
        if N < 32:
            thr = 0
        else:
            thr = 0.3936 + 0.1829 * (np.log(N) / np.log(2))
        return thr

    elif thr_select == 'rigrsure':
        # #风险阈值
        # sx = np.sort(abs(self.data))
        # sx2 = np.square(sx)
        # N1 = np.repeat(N-2*[i for i in range(0,N)],1)
        thr = rigrsure(data,N)
        return thr

# 本函数实现输入波形的小波阈值降噪，配合上方定义的sgn使用
# 参考https://zhuanlan.zhihu.com/p/157540476
def wavelet_noising(data,wave = "coif4",thr_select = "rigrsure",thr_way = "soft"):
    # data = data.values.T.tolist()  # 将np.ndarray()转为列表
    # w = pywt.Wavelet(wave)#选择小波基
    # print(w)

    # [ca6, cd6, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=6)  # 6层小波分解,低频分量，高频分量，...，

    coffe = pywt.wavedec(data, wave, level=6)

    # 低频分量
    ca6 = coffe[0]
    cd_out_list = []
    cd_out_list.append(ca6)

    # 高频分量
    cd1 = coffe[6]
    # print("----len-----",len(cd1))
    Cd1 = np.array(cd1)
    # 用第一层的高频系数估算噪声的方差
    abs_cd1 = np.abs(Cd1)
    # 计算均值
    median_cd1 = np.median(abs_cd1)
    sigma =(1.0 / 0.6745) * median_cd1
    # print("sigma:",sigma)


    thr = thrselect(data, thr_select)
    # 根据选择的阈值计算公式计算阈值
    lamda = thr*sigma

    for i in range(1, len(coffe)):
       cd = coffe[i]
       if thr_way == "new":
           # 改进小波阈值去噪方法在电机电流信号处理中的应用_马欢.pdf
           for j in range(len(cd)):
               if (abs(cd[j]) >= lamda):
                   b = 1
                   cd[j] = sgn(cd[j]) * ( abs(cd[j]) - lamda * math.exp(b*(lamda - abs(cd[j]))) )
                   # cd[j] = sgn(cd[j]) * (abs(cd[j]) - lamda / np.log2(i + 1))
               else:
                   cd[j] = 0
           cd_out_list.append(cd)
       elif thr_way == "hard":
           # snr:61.37584995,rmse:0.00011713

           for j in range(len(cd)):
               if (abs(cd[j]) >= lamda):
                   cd[j] = cd[j]
               else:
                   cd[j] = 0
           cd_out_list.append(cd)

       else:
           for j in range(len(cd)):
               if (abs(cd[j]) >= lamda):
                   cd[j] = sgn(cd[j]) * ( abs(cd[j]) - lamda)
               else:
                   cd[j] = 0
           cd_out_list.append(cd)
           # y = pywt.threshold(cd, lamda, thr_way)
           # cd_out_list.append(y)

    recoeffs = pywt.waverec(cd_out_list, wave)


    return recoeffs

# Draw 输入两条线，在同一个图里以不同色展示，在图中加一个小窗口，用于放大细节
def Draw(line1,line2):
    MAX_EPISODES = len(line2)
    x_axis_data = []
    for l in range(MAX_EPISODES):
        x_axis_data.append(l)
    # 没有添加线的x坐标信息，没有实现自动窗口显示范围自动获取
    # 全文参考https://blog.csdn.net/wulishinian/article/details/106668011
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_axis_data,line1, color='#4169E1', alpha=0.8, label='raw data')
    ax.plot(x_axis_data,line2, color='#B84D37', alpha=0.8, label='denoise data')
    ax.legend()
    ax.set_xlabel('Times')
    ax.set_ylabel('Value')

    # 画子图框框
    axins = ax.inset_axes((0.4, 0.1, 0.5, 0.3))  # 坐标，大小
    # axins = ax.inset_axes((0.55, 0.4, 0.4, 0.3))  # 坐标，大小
    axins.plot(x_axis_data,line1, color='#4169E1', alpha=0.8, label='raw data')
    axins.plot(x_axis_data,line2, color='#B84D37', alpha=0.8, label='denoisinf data')
    # 设置放大区间
    zone_left = 100
    zone_right = 200

    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0  # x轴显示范围的扩展比例
    y_ratio = 0.05  # y轴显示范围的扩展比例

    # X轴的显示范围
    xlim0 = x_axis_data[zone_left] - (x_axis_data[zone_right] - x_axis_data[zone_left]) * x_ratio
    xlim1 = x_axis_data[zone_right] + (x_axis_data[zone_right] - x_axis_data[zone_left]) * x_ratio

    # Y轴的显示范围
    y = np.hstack((line1[zone_left:zone_right], line2[zone_left:zone_right]))
    ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
    ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    # 原图中画方框
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    # 画两条线
    xy = (xlim0, ylim0)
    xy2 = (xlim0, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)
    xy = (xlim1, ylim0)
    xy2 = (xlim1, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    plt.show()

def compute_snr(org_signal, final_signal):
    """
    信噪比:信噪比越大越好
    均方根误差:均方根误差越小越好，越小去噪效果越好
    :param org_signal:原始信号
    :param final_signal:降噪后的信号
    :return: 信噪比,均方根误差
    """
    clean = np.array(final_signal)
    org_signal = np.array(org_signal)
    sigPower = sum(abs(clean) ** 2)  # 求出信号功率
    noisePower = sum(abs(org_signal - clean) ** 2)   # 求出噪声功率
    SNR_10 = 10 * np.log10(sigPower / noisePower)
    # SNR_10 = (sigPower / noisePower)
    return SNR_10

def compute_mse(org_signal, final_signal):
    """
    计算均方根误差:均方根误差越小越好，越小去噪效果越好
    :param org_signal:原始信号
    :param final_signal:降噪后的信号
    :return: 均方根误差
    """
    data = np.array(org_signal)
    final_signal = np.array(final_signal)
    rmse = np.sqrt(np.mean(np.square(data - final_signal)))
    return rmse

def get_eff(org_signal,wave,thr_select,thr_way):
    final_signal = wavelet_noising(org_signal,wave,thr_select,thr_way)
    SNR = round(compute_snr(org_signal,final_signal),8)
    RMSE = round(compute_mse(org_signal,final_signal),8)
    print('wave:{}, thr_select:{}, thr_way:{}'.format(wave, thr_select,thr_way))
    print('snr:{},rmse:{}'.format(SNR, RMSE))
    print()
    return SNR,RMSE,final_signal

if __name__ == '__main__':
    path = 'D:\\1HLB\\LY202112\\1223r1\\1223r1-ch6.D01'
    # encoding = get_encoding(path)
    # print("文件编码：",encoding)
    curveData, mesage, = get_data(path)
    # print(mesage) # 打印文件头的文本信息
    # print(curveData[:3])  # 看看数据对不对

    '''
    #小波降噪 不知道怎么降噪的降噪
    # level,datarec = DWT_Threshold(curveData,500)
    # print("maximum level is ",level)
    '''

    begin = 0   # 定义波形起始位置
    wave = "coif4"
    thr_select = "minimaxi"
    thr_way = "new"


    # curveData[26300:27300] G波
    SNR, RMSE , data_denoise= get_eff(curveData[:], wave, thr_select,thr_way)

    # # 获得ppt表格
    for i in ["db4","sym4","coif4"]:
        for j in ["hard","soft","new"]:
            SNR, RMSE, data_denoise = get_eff(curveData, i, thr_select, j)

    # 傅里叶变换，第一个参数调用plot可得到频谱，第二个不知道什么用
    # r1,rp = fft_Trans(curveData,begin)
    # c1,cp = fft_Trans(data_denoising,begin)

    # 小波连续变换,直接获得时频图
    # wavelet_Trans(curveData[67200:68200])
    # wavelet_Trans(data_denoise[67200:68200])

    # 画降噪前后的对比图
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(curveData[142000:143000], color='#4169E1', alpha=0.8, label='raw data')
    # ax.plot(data_denoising[142000:143000], color='#B84D37', alpha=0.8, label='denoise data')
    # ax.legend()
    # ax.set_xlabel('Times')
    # ax.set_ylabel('Value')
    # plt.show()

    # 获得ppt的图，降噪前后的对比图
    # Draw(curveData[50000:53000],data_denoising[50000:53000])
    data_denoising = wavelet_noising(curveData, wave, thr_select, thr_way)  # 调用函数进行小波阈值去噪,得到相同的x轴
    Draw(curveData[67200:68100], data_denoising[67200:68100])
    # Draw(curveData[26300:27300],data_denoising[26300:27300])
    # Draw(r1,c1)
    # Draw(curveData, data_denoising)

    # snr = compute_snr(curveData[142000:143000], data_denoising[142000:143000])
    # rmse = compute_mse(curveData[142000:143000], data_denoising[142000:143000])
    # snr: 46.08649773142977, rmse: 0.0009666971145705327 1
    # snr: 44.93288888639356, rmse: 0.0011036363359736617 6

    # snr = compute_snr(curveData, data_denoising)
    # rmse = compute_mse(curveData, data_denoising)
    # # snr: 49.58975243385232,rmse:0.0009484055570491831 1
    # # snr: 48.76733307810148, rmse: 0.0010425489885609216 2
    # # snr: 48.40261522592981, rmse: 0.0010871756825311734 6




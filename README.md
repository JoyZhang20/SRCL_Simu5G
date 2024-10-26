##### 版本

win11

OMNet++6.1.0

INET4.5.2

Simu5G

##### 安装（依次安装OMNet++,Inet，simu5G）

###### 安装OMNet++6.1.0

1.在[OMNet++官网](https://omnetpp.org/download/)下载安装包后解压到本地

2.执行mingwenv.cmd

3.在命令行中输入./configure

4.输入make

###### 安装INET4.5.2

1.创建一个目录作为OMNET++的workspace

2.从[INET Framework官网](https://inet.omnetpp.org/Download.html)下载安装包后解压到第1步中的workspace

3.打开OMNeT++ 6.1 IDE，File-Import-Existing Projects into Workspace选择第2步中的INET解压目录

4.打开OMNeT++ 6.1 Shell，使用下列命令编译inet4.5

```bash
cd workspace/inet#指向自己的目录
make -j4#多线程加速
```

或者直接在IDE中选择Project-Build All
###### 安装Simu5G

1.从[Simu5G仓库](https://inet.omnetpp.org/Download.html)下载代码后解压到上面的workspace

3.打开OMNeT++ 6.1 IDE，File-Import-Existing Projects into Workspace选择上面第1步中的Simu5G解压目录

4.打开OMNeT++ 6.1 Shell，使用下列命令编译Simu5G

```bash
cd workspace/simu5g#指向自己的目录
make -j4#多线程加速
```

或者直接在IDE中选择Project-Build All

###### 可能出现的问题



###### 验证

打开Simu5G/simulations/NR/mec/requestResponseApp/omnetpp.ini后，点击左上角绿色的run按钮

###### 更新

将我们的requestResponseApp替换到Simu5G/simulations/NR/mec

将我们的VirtualisationInfrastructureManager替换到Simu5G/src/nodes/mec

###### 运行

运行python程序

python3 server.py

运行Simu5G/simulations/NR/mec/requestResponseApp/omnetpp.ini

###### 观察结果

点击Simu5G/simulations/NR/mec/requestResponseApp/results/MultiMec/MultiMec--0.sca构建MultiMec-.anf，在IDE中分析或导出csv后分析（使用MultiMecHost_delay.ue[0].app[1]过滤得到用户关键指标）。

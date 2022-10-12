# alphazero-aichess
 本项目为
<a href="https://github.com/tensorfly-gpu/aichess.git">https://github.com/tensorfly-gpu/aichess.git
复现，用途为个人学习记录，并后续打算在其基础上进行改进

 原作者：tensorfly-gpu 一心炼银

#


## 网络结构
<img src="/imgs/structure.png" alt="structure" title="structure">

## 环境配置
pytorch 11.x + cuda 

注：源项目中有paddle实现已经被注释，具体可访问文档开头项目地址

## 训练
<p> cd到项目目录下 </p>

`python collect.py` (支持分布式)

`python train.py`

## 运行

`python play_with_ai.py` (输入4位数字，连两位代表原本棋子位置，后两位代表移动到的位置)

或

`python UIplay.py`

## 感谢

tensorfly-gpu 一心炼银 (https://github.com/tensorfly-gpu)














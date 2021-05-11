#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.yolo3 import YoloBody

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = YoloBody([[1,2,3],[2,3,4],[3,4,5]], 80).to(device)
    summary(m, input_size=(3, 416, 416))

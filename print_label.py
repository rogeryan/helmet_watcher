# -*- coding: utf-8 -*-
"""
打印出数据集标记XML中所有目标名称
"""
from configuration import PASCAL_VOC_DIR
import xml.etree.ElementTree as ET
import os

lables = set()
def get_label(filename):
    tree = ET.parse(filename)
    objs = tree.findall('object')
    
    for obj in objs:
        lables.add(obj.find('name').text)


if __name__ == '__main__':
    ano_dir = PASCAL_VOC_DIR + "Annotations"
    filename_list = os.listdir(ano_dir)
    for filename in filename_list:
        filepath = os.path.join(ano_dir, filename)
        get_label(filepath)
    print(lables)


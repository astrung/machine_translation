#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chương trình chuyển đổi từ Tiếng Việt có dấu sang Tiếng Việt không dấu
"""

import re
import unicodedata


def no_accent_vietnamese(s):
    # s = s.decode('utf-8')
    s = re.sub(u'Đ', 'D', s)
    s = re.sub(u'đ', 'd', s)
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore')


if __name__ == '__main__':
    # print(no_accent_vietnamese("Việt Nam Đất Nước Con Người"))
    # print(no_accent_vietnamese("Welcome to Vietnam !"))
    # print(no_accent_vietnamese("VIỆT NAM ĐẤT NƯỚC CON NGƯỜI"))
    with open('VNTQcorpus-small.txt', encoding='utf-8') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [str(x.strip()) for x in content]
    no_accent_content = []
    print("Strip accent")
    for line in content:
        no_accent_content.append(no_accent_vietnamese(line))
    file = open("VNTQcorpus-small_no_accent.txt", "w")
    print("write file")
    for line in no_accent_content:
        print(line)
        file.write(line.decode("utf-8")+"\n")
        # file.write(bytes("\n"))
    file.close()

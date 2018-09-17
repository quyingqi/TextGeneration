#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
with open('data/val.de', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        line = line.encode('utf-8')
        print(line[:])

with open('data/test2016.de-tmp') as f:
    for line in f:
        line = line.strip()
        line = line[2:-1]
        if line:
            print(line)
'''
with open('data/val.en') as f:
    for line in f:
        line = line.strip()
        print(line[0])

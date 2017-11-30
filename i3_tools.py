import PhotoScan

import importlib

import i3_tools.trans_chunck

trans_chunck = importlib.reload(i3_tools.trans_chunck)


def init():
    # import because photoscan to not like scrips in multiple files
    print('i3 tools added')
    PhotoScan.app.addMenuItem("i3 tools/Transform Chunck by Matrix", open_trans_chunk)


def open_trans_chunk():
    print('openwindow')

    trans_chunck.trans_chunck_init()


init()

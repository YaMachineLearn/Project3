
import theano
import numpy as np

LABEL_NUM = 48

LABEL_LIST = [
    "aa" ,
    "ae" ,
    "ah" ,
    "ao" ,
    "aw" ,
    "ax" ,
    "ay" ,
    "b"  ,
    "ch" ,
    "cl" ,
    "d"  ,
    "dh" ,
    "dx" ,
    "eh" ,
    "el" ,
    "en" ,
    "epi",
    "er" ,
    "ey" ,
    "f"  ,
    "g"  ,
    "hh" ,
    "ih" ,
    "ix" ,
    "iy" ,
    "jh" ,
    "k"  ,
    "l"  ,
    "m"  ,
    "ng" ,
    "n"  ,
    "ow" ,
    "oy" ,
    "p"  ,
    "r"  ,
    "sh" ,
    "sil",
    "s"  ,
    "th" ,
    "t"  ,
    "uh" ,
    "uw" ,
    "vcl",
    "v"  ,
    "w"  ,
    "y"  ,
    "zh" ,
    "z"
]

DICT_LABEL_NUM = {
    "aa"    :0,
    "ae"    :1,
    "ah"    :2,
    "ao"    :3,
    "aw"    :4,
    "ax"    :5,
    "ay"    :6,
    "b"     :7,
    "ch"    :8,
    "cl"    :9,
    "d"     :10,
    "dh"    :11,
    "dx"    :12,
    "eh"    :13,
    "el"    :14,
    "en"    :15,
    "epi"   :16,
    "er"    :17,
    "ey"    :18,
    "f"     :19,
    "g"     :20,
    "hh"    :21,
    "ih"    :22,
    "ix"    :23,
    "iy"    :24,
    "jh"    :25,
    "k"     :26,
    "l"     :27,
    "m"     :28,
    "ng"    :29,
    "n"     :30,
    "ow"    :31,
    "oy"    :32,
    "p"     :33,
    "r"     :34,
    "sh"    :35,
    "sil"   :36,
    "s"     :37,
    "th"    :38,
    "t"     :39,
    "uh"    :40,
    "uw"    :41,
    "vcl"   :42,
    "v"     :43,
    "w"     :44,
    "y"     :45,
    "zh"    :46,
    "z"     :47
}

DICT_LABEL_LABEL = {
    "aa"    :"aa",
    "ae"    :"ae",
    "ah"    :"ah",
    "ao"    :"aa",
    "aw"    :"aw",
    "ax"    :"ah",
    "ay"    :"ay",
    "b"     :"b",
    "ch"    :"ch",
    "cl"    :"sil",
    "d"     :"d",
    "dh"    :"dh",
    "dx"    :"dx",
    "eh"    :"dh",
    "el"    :"l",
    "en"    :"n",
    "epi"   :"sil",
    "er"    :"er",
    "ey"    :"ey",
    "f"     :"f",
    "g"     :"g",
    "hh"    :"hh",
    "ih"    :"ih",
    "ix"    :"ih",
    "iy"    :"iy",
    "jh"    :"jh",
    "k"     :"k",
    "l"     :"l",
    "m"     :"m",
    "ng"    :"ng",
    "n"     :"n",
    "ow"    :"ow",
    "oy"    :"oy",
    "p"     :"p",
    "r"     :"r",
    "sh"    :"sh",
    "sil"   :"sil",
    "s"     :"s",
    "th"    :"th",
    "t"     :"t",
    "uh"    :"uh",
    "uw"    :"uw",
    "vcl"   :"sil",
    "v"     :"v",
    "w"     :"w",
    "y"     :"y",
    "zh"    :"sh",
    "z"     :"z"
}

def labelToIndex(label):
    return DICT_LABEL_NUM[label]

def labelsToIndices(labels):
    list = []
    for lb in labels:
        list.append(DICT_LABEL_NUM[lb])
    return np.asarray(list, dtype=theano.config.floatX)

def indexToLabel(index):
    return LABEL_LIST[index]

def labelToLabel(label):
    #48 to 39
    return DICT_LABEL_LABEL[label]

def labelToList(label):
    #to [0,...,0,1,0,...0]
    l = [0] * 48
    l[ DICT_LABEL_NUM[label] ] = 1
    return l

def labelToArray(label):
    #better: should be named "labelsToArrays(labelList)"
    #   or "labelListToArrayList"
    list = []
    for lb in label:
        l = [0] * 48
        l[ DICT_LABEL_NUM[lb] ] = 1
        list.append(l)
    return np.asarray(list, dtype=theano.config.floatX)

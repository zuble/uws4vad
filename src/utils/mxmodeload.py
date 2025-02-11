from os import path as osp , makedirs as makedirs
import glob, sys, time, argparse, gc, re
import numpy

# from mxnet import np, npx, nd
# from mxnet.gluon import nn
# npx.set_np()

import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model

## https://cv.gluon.ai/model_zoo/action_recognition.html
classes = 400
model_name = "i3d_inceptionv1_kinetics400"#"slowfast_4x16_resnet50_kinetics400"
hashtag = "81e0be10" #"9d650f51"
context = mx.cpu()

## as each crop from each clip is forwarded as one video itself, num_segments,num_crop is 1
## https://github.com/dmlc/gluon-cv/blob/567775619f3b97d47e7c360748912a4fd883ff52/gluoncv/model_zoo/action_recognition/i3d_resnet.py#L619
net = get_model(
    name=model_name, 
    nclass=classes, 
    pretrained=hashtag,
    feat_ext=True, 
    num_segments=1, 
    num_crop=1, 
    ctx=context
)
#net.cast("float32")
#net.collect_params().reset_ctx(context)
#net.hybridize(static_alloc=True, static_shape=True)
print("Successfully built model {}".format(model_name))


x = nd.ones((1,3,64,224,224))
print(net.summary(x))


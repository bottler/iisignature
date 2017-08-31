import tensorflow as tf

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature

from iisignature_tensorflow import Sig, LogSig

def trySig():
    sess=tf.Session()
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    out = Sig(a,b)
    grad = tf.gradients(out,a)[0]

    path=[[2,3],[3,5],[4,5]]
    m=2

    out_vals=(sess.run([out,grad], {a: path, b: m}))
    print (out_vals[0])
    print(out_vals[1])
#trySig()

def tryLogSig():
    for expanded in (False, True):
        sess=tf.Session()
        path=[[2,3],[3,5],[4,5]]
        m=2
        s=iisignature.prepare(2,m)
        a = tf.placeholder(tf.float32)
        out= LogSig(a,s,"x") if expanded else LogSig(a,s)
        grad = tf.gradients(tf.reduce_sum(out),a)[0]
        #grad = tf.gradients(out,a)[0]
        
        out_vals=(sess.run([out,grad], {a: path}))
        print (out_vals[0])
        print(out_vals[1])
tryLogSig()

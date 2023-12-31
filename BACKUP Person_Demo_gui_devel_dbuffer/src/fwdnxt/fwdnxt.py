import sys
from ctypes import *
import numpy as np
from numpy.ctypeslib import as_ctypes
from numpy.ctypeslib import ndpointer
#f = CDLL("./src/fwdnxt/libfwdnxt_dbuffer.so")

# Allows None to be passed instead of a ndarray


def wrapped_ndptr(*args, **kwargs):
    base = ndpointer(*args, **kwargs)

    def from_param(cls, obj):
        if obj is None:
            return obj
        return base.from_param(obj)
    return type(base.__name__, (base,), {
                'from_param': classmethod(from_param)})


FloatNdPtr = wrapped_ndptr(dtype=np.float32, flags='C_CONTIGUOUS')


class FWDNXT:
    def __init__(self):
        self.userobjs = {}

        self.ie_create = f.ie_create
        self.ie_create.restype = c_void_p

        self.handle = f.ie_create()

        self.ie_loadmulti = f.ie_loadmulti
        self.ie_loadmulti.argtypes = [c_void_p, POINTER(c_char_p), c_int]
        self.ie_loadmulti.restype = c_void_p

        self.ie_compile = f.ie_compile
        self.ie_compile.restype = c_void_p

        self.ie_init = f.ie_init
        self.ie_init.restype = c_void_p

        self.ie_free = f.ie_free
        self.ie_free.argtypes = [c_void_p]

        self.ie_setflag = f.ie_setflag

        self.ie_getinfo = f.ie_getinfo

        self.ie_run = f.ie_run
        self.ie_run.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(
            c_ulonglong), POINTER(POINTER(c_float)), POINTER(c_ulonglong)]

        self.ie_putinput = f.ie_putinput
        self.ie_putinput.argtypes = [c_void_p, POINTER(
            POINTER(c_float)), POINTER(c_ulonglong), c_long]

        self.ie_getresult = f.ie_getresult
        self.ie_getresult.argtypes = [c_void_p, POINTER(
            POINTER(c_float)), POINTER(c_ulonglong), c_void_p]

        self.ie_read_data = f.ie_read_data
        self.ie_read_data.argtypes = [c_void_p, c_ulonglong, ndpointer(
            c_float, flags="C_CONTIGUOUS"), c_ulonglong, c_int]

        self.ie_write_data = f.ie_write_data
        self.ie_write_data.argtypes = [c_void_p, c_ulonglong, ndpointer(
            c_float, flags="C_CONTIGUOUS"), c_ulonglong, c_int]

        self.ie_write_weights = f.ie_write_weights
        self.ie_write_weights.argtypes = [c_void_p, ndpointer(
            c_float, flags="C_CONTIGUOUS"), c_int, c_int]

        self.ie_run_sim = f.ie_run_sim
        self.ie_run_sim.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(
            c_ulonglong), POINTER(POINTER(c_float)), POINTER(c_ulonglong)]

        self.thnets_run_sim = f.thnets_run_sim
        self.thnets_run_sim.argtypes = [c_void_p, POINTER(POINTER(c_float)), POINTER(
            c_ulonglong), POINTER(POINTER(c_float)), POINTER(c_ulonglong), c_bool]

        # Training of linear layer
        self.trainlinear_start = f.ie_trainlinear_start
        self.trainlinear_start.argtypes = [c_void_p, c_int, c_int, c_int, ndpointer(
            c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int, c_int, c_int, c_float]

        self.trainlinear_data = f.ie_trainlinear_data
        self.trainlinear_data.argtypes = [
            c_void_p, FloatNdPtr, FloatNdPtr, c_int]

        self.trainlinear_step_sw = f.ie_trainlinear_step_sw
        self.trainlinear_step_sw.argtypes = [c_void_p]

        self.trainlinear_step_float = f.ie_trainlinear_step_float
        self.trainlinear_step_float.argtypes = [c_void_p]

        self.trainlinear_step = f.ie_trainlinear_step
        self.trainlinear_step.argtypes = [c_void_p, c_int]

        self.trainlinear_get = f.ie_trainlinear_get
        self.trainlinear_get.argtypes = [c_void_p, ndpointer(
            c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS")]

        self.trainlinear_getY = f.ie_trainlinear_getY
        self.trainlinear_getY.argtypes = [
            c_void_p, ndpointer(c_float, flags="C_CONTIGUOUS")]

        self.trainlinear_end = f.ie_trainlinear_end
        self.trainlinear_end.argtypes = [c_void_p]

    def TrainlinearStart(self, batchsize, A, b, Ashift,
                         Xshift, Yshift, Ygshift, rate):
        self.trainlinear_start(
            self.handle, A.shape[1], A.shape[0], batchsize, A, b, Ashift, Xshift, Yshift, Ygshift, rate)

    def TrainlinearData(self, X, Y, idx):
        self.trainlinear_data(self.handle, X, Y, idx)

    def TrainlinearStep(self, idx):
        self.trainlinear_step(self.handle, idx)

    def TrainlinearStepSw(self):
        self.trainlinear_step_sw(self.handle)

    def TrainlinearStepFloat(self):
        self.trainlinear_step_float(self.handle)

    def TrainlinearGet(self, A, b):
        self.trainlinear_get(self.handle, A, b)

    def TrainlinearGetY(self, Y):
        self.trainlinear_getY(self.handle, Y)

    def TrainlinearEnd(self):
        self.trainlinear_end(self.handle)

    def Loadmulti(self, bins):
        b = (c_char_p * len(bins))()
        for i in range(len(bins)):
            b[i] = bytes(bins[i], 'utf-8')
        self.ie_loadmulti(self.handle, b, len(bins))

    # compile a network and produce .bin file with everything that is needed
    # to execute
    def Compile(self, image, modeldir, outfile,
                numcard=1, numclus=1, nlayers=-1):
        self.swoutsize = (c_ulonglong * 16)()
        self.noutputs = c_int()
        self.ie_compile(self.handle, bytes(image, 'ascii'), bytes(modeldir, 'ascii'),
                        bytes(outfile, 'ascii'), self.swoutsize, byref(self.noutputs), numcard, numclus, nlayers, False)
        if self.noutputs.value == 1:
            return self.swoutsize[0]
        ret = ()
        for i in range(self.noutputs.value):
            ret += (self.swoutsize[i],)
        return ret
    # returns the context obj

    def get_handle(self):
        return self.handle
    # initialization routines for FWDNXT inference engine

    def Init(self, infile, bitfile, cmem=None):
        self.outsize = (c_ulonglong * 16)()
        self.noutputs = c_int()
        self.handle = self.ie_init(self.handle, bytes(bitfile, 'ascii'), bytes(
            infile, 'ascii'), byref(self.outsize), byref(self.noutputs), cmem)
        if self.noutputs.value == 1:
            return self.outsize[0]
        ret = ()
        for i in range(self.noutputs.value):
            ret += (self.outsize[i],)
        return ret

    # Free FPGA instance
    def Free(self):
        self.ie_free(self.handle)
        self.handle = c_void_p()

    # Set flags for the compiler
    def SetFlag(self, name, value):
        rc = self.ie_setflag(self.handle, bytes(
            name, 'ascii'), bytes(value, 'ascii'))
        if rc != 0:
            raise Exception(rc)

    # Get various info about the inference engine
    def GetInfo(self, name):
        if name == 'hwtime':
            return_val = c_float()
        else:
            return_val = c_int()
        rc = self.ie_getinfo(self.handle, bytes(
            name, 'ascii'), byref(return_val))
        if rc != 0:
            raise Exception(rc)
        return return_val.value

    def params(self, images):
        if isinstance(images, np.ndarray):
            return byref(images.ctypes.data_as(POINTER(c_float))
                         ), pointer(c_ulonglong(images.size))
        elif isinstance(images, tuple):
            cimages = (POINTER(c_float) * len(images))()
            csizes = (c_ulonglong * len(images))()
            for i in range(len(images)):
                cimages[i] = images[i].ctypes.data_as(POINTER(c_float))
                csizes[i] = images[i].size
            return cimages, csizes
        else:
            raise Exception('Input must be ndarray or tuple to ndarrays')

    # Run inference engine. It does the steps sequentially. putInput, compute,
    # getResult

    def Run(self, images, result):
        imgs, sizes = self.params(images)
        r, rs = self.params(result)
        rc = self.ie_run(self.handle, imgs, sizes, r, rs)
        if rc != 0:
            raise Exception(rc)

    # Put an input into shared memory and start FWDNXT hardware
    def PutInput(self, images, userobj):
        userobj = py_object(userobj)
        key = c_long(addressof(userobj))
        self.userobjs[key.value] = userobj
        if images is None:
            imgs, sizes = None, None
        else:
            imgs, sizes = self.params(images)
        rc = self.ie_putinput(self.handle, imgs, sizes, key)
        if rc == -99:
            return False
        if rc != 0:
            raise Exception(rc)
        return True

    # Get an output from shared memory. If opt_blocking was set then it will
    # wait FWDNXT hardware
    def GetResult(self, result):
        userobj = c_long()
        r, rs = self.params(result)
        rc = self.ie_getresult(self.handle, r, rs, byref(userobj))
        if rc == -99:
            return None
        if rc != 0:
            raise Exception(rc)
        retuserobj = self.userobjs[userobj.value]
        del self.userobjs[userobj.value]
        return retuserobj.value

    # Run software inference engine emulator
    def Run_sw(self, images, result):
        imgs, sizes = self.params(images)
        r, rs = self.params(result)
        rc = self.ie_run_sim(self.handle, imgs, sizes, r, rs)
        if rc != 0:
            raise Exception(rc)

    # Run model with thnets
    def Run_th(self, image, result):
        imgs, sizes = self.params(images)
        r, rs = self.params(result)
        rc = self.thnets_run_sim(self.handle, imgs, sizes, r, rs)
        if rc != 0:
            raise Exception(rc)

    # read data from an address in shared memory
    def ReadData(self, addr, data, card):
        self.ie_read_data(self.handle, addr, data,
                          data.size * sizeof(c_float), card)

    # write data to an address in shared memory
    def WriteData(self, addr, data, card):
        self.ie_write_data(self.handle, addr, data,
                           data.size * sizeof(c_float), card)

    def WriteWeights(self, weight, node):
        self.ie_write_weights(self.handle, weight, weight.size, node)

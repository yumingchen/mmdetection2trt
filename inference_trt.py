import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import ctypes
import utils
import time

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.3


class ModelTRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_path):
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
        # Deserialize the engine from file
        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        # self.host_inputs = []
        # self.cuda_inputs = []
        # self.host_outputs = []
        # self.cuda_outputs = []
        # self.bindings = []
        # self.input_names = ['input_0']
        # self.output_names = ['num_detections', 'boxes', 'scores', 'classes']

    def infer_video(self, input_image):
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, tuple([1, 3, input_h, input_w]))
            shape = tuple(self.context.get_binding_shape(idx))
            if self.engine.binding_is_input(binding):
                shape = tuple([1, 3, input_h, input_w])
                shape = 1*3*input_h*input_w
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(shape, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], self.stream)
        # Run inference.
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], self.stream) for i in range(len(host_outputs))]
        # Synchronize the stream
        self.stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

        outputs = tuple(host_outputs)
        if len(outputs) == 1:
            outputs = outputs[0]
        result = list(outputs)
        # result[1] [[[x1,y1,x2,y2]...]], shape=(1,100, 4)
        return result

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()


if __name__ == '__main__':
    PLUGIN_LIBRARY = "/home/cym/programfiles/amirstan_plugin/build/lib/libamirstan_plugin.so"
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_path = '/home/cym/CYM/output/cascade_rcnn/epoch_37_pth.engine'
    cascade_rcnn_trt = ModelTRT(engine_path)
    image_paths = [
        '/home/cym/CYM/dataset/comp2_coco/images/0afba5d6_ea2a_4490_9a01_74f1f367d1e1.JPG',
        '/home/cym/CYM/dataset/comp2_coco/images/0a7827d6_484d_464a_968e_3b16570dc0ce.JPG',
        '/home/cym/CYM/dataset/comp2_coco/images/b7a03e67_33a9_4fe7_bd1a_e30cd0f62f64.jpg',
        '/home/cym/CYM/dataset/comp2_coco/images/d8d781fb_8f1b_42d6_89a3_1f1a96293f66.JPG',
        '/home/cym/CYM/dataset/comp2_coco/images/2286849b_7a57_4ab2_bd4f_f057e1d81d23.jpg',
        '/home/cym/CYM/dataset/comp2_coco/images/71ad51d7_571b_456e_9b47_5c63fb0154d7.jpg',
        '/home/cym/CYM/dataset/comp2_coco/images/53d6f842_43cf_4cb4_88a1_4fc49574c2e1.jpg',
        '/home/cym/CYM/dataset/comp2_coco/images/3ea40866_eb42_4dee_b130_b35644fa0970.JPG',
        '/home/cym/CYM/dataset/comp2_coco/images/25d11b90_c728_4763_8ad9_007c638c5b7d.JPG',
        '/home/cym/CYM/dataset/comp2_coco/images/be8bd121_03ac_445b_80ed_537b8fcb8639.jpg',
        '/home/cym/CYM/dataset/comp2_coco/images/d974dcf7_ed0b_4246_826f_9d152b0e8d8d.JPG',
        '/home/cym/CYM/dataset/comp2_coco/images/a9ca65dc_92de_4136_bf9f_64710bf302ec.jpg',
        '/home/cym/CYM/dataset/comp2_coco/images/a2551484_7adc_45c8_b58c_4b1b67217ff2.jpg',
        '/home/cym/CYM/dataset/comp2_coco/images/429e062a_d5a4_48a2_be8d_fb3e3d8f7e53.jpg',
        '/home/cym/CYM/dataset/comp2_coco/images/f48f108d_3482_45c3_8657_0e1aadbec933.jpg',
        '/home/cym/CYM/dataset/comp2_coco/images/b258a66c_277c_4a0f_b4c4_0b57faaf6ba6.jpg',
    ]
    for i in range(len(image_paths)):
        if os.path.exists(image_paths[i]):
            print('Reading the image [{}]'.format(image_paths[i]))
        else:
            print('The image [{}] is not existed!'.format(image_paths[i]))
        start_all = time.time()
        input_image, image_raw, origin_h, origin_w, input_h, input_w, image_bgr = \
            utils.preprocess_image(image_paths[i], input_size=(1333, 800))
        end_pre = time.time()
        result = cascade_rcnn_trt.infer_video(input_image)
        end_infer = time.time()
        utils.postprocess_image(result, image_raw, origin_h, origin_w, input_h, input_w, image_bgr)
        end_post = time.time()
        print('preprocess:[{}s] inference:[{}s] postprocess:[{}s]'.format(end_pre-start_all, end_infer-end_pre, end_post-end_infer))
        print('all time:[{}s]'.format(end_post-start_all))
    cascade_rcnn_trt.destroy()









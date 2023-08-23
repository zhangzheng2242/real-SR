from rknn.api import RKNN
import os
import cv2
import numpy as np
if __name__ == '__main__':
    platform = 'rk3588'

    '''step 1: create RKNN object'''
    rknn = RKNN()

    '''step 2: load the .rknn model'''
    rknn.config(target_platform='rk3588')
    print('--> Loading model')
    ret = rknn.load_onnx('./vapsrS-x2.onnx',input_size_list=[[1, 400, 400, 3]])
    if ret != 0:
        print('load model failed')
        exit(ret)
    print('done')

    '''step 3: building model'''
    print('-->Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('build model failed')
        exit()
    print('done')

    # '''step 4: export and save the .rknn model'''
    # OUT_DIR = 'rknn_models'
    # RKNN_MODEL_PATH = './{}/vapsrS-x2.rknn'.format(OUT_DIR)
    # if not os.path.exists(OUT_DIR):
    #     os.mkdir(OUT_DIR)
    # print('--> Export RKNN model: {}'.format(RKNN_MODEL_PATH))
    # ret = rknn.export_rknn(RKNN_MODEL_PATH)
    # if ret != 0:
    #     print('Export rknn model failed.')
    #     exit(ret)
    # print('done')

    ####调用init_rubtime接口初始化运行环境
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')


    ####调用inference接口对输入进行处理
    IMG_PATH ='./2.png'
    img = cv2.imread(IMG_PATH)
    # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    img = img/255.0
    print('--> Running model')
    img = np.expand_dims(img,axis=0)
    outputs = rknn.inference(inputs=[img])
    
    outputs = np.dot(255.0,outputs[0])

    outputs = np.squeeze(outputs, axis=0)
    
    outputs = np.clip(outputs,0.0,255.0).astype(np.uint8).transpose([1,2,0])

    outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./results/" + "test_simulator.png", outputs)
    print('save image')
    '''step 5: release the model'''
    rknn.release()
    print('rknn.release() done')
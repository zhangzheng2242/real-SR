from rknn.api import RKNN
import os

if __name__ == '__main__':
    platform = 'rk3588'

    '''step 1: create RKNN object'''
    rknn = RKNN()

    '''step 2: load the .onnx model'''
    rknn.config(target_platform='rk3588')
    print('--> Loading model')
    ret = rknn.load_onnx('vapsrS-x2.onnx')
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

    '''step 4: export and save the .rknn model'''
    OUT_DIR = 'rknn_models'
    RKNN_MODEL_PATH = './{}/vapsrS-x2.rknn'.format(OUT_DIR)
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    print('--> Export RKNN model: {}'.format(RKNN_MODEL_PATH))
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print('Export rknn model failed.')
        exit(ret)
    print('done')

    '''step 5: release the model'''
    rknn.release()
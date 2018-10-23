#include <jni.h>
#include <string>
#include "network.h"
#include <string>
#include <android/log.h>
#include <fstream>
#include "network.h"

#define  LOG_TAG    "debug::native-lib.cpp"

using namespace std;

int my_counter __attribute__((aligned(8)));

extern "C"
JNIEXPORT jlong JNICALL
Java_edu_northeastern_nliu_c3d_NativeEntryInterface_buildNetwork(JNIEnv *env, jobject instance) {

    network *networkObj = new network("c3d");
    //add layers
    Layer *conv1 = new ThreeDConvLayer("conv1", 3, 3, 3, 3, 64, new int[5]{1, 1, 1, 1, 1},
                                       new int[5]{1, 1, 1, 1, 1});
    Layer *maxpool1 = new MaxThreeDPooling("maxpool1", new int[5]{1, 1, 2, 2, 1},
                                           new int[5]{1, 1, 2, 2, 1});
    Layer *conv2 = new ThreeDConvLayer("conv2", 3, 3, 3, 64, 128, new int[5]{1, 1, 1, 1, 1},
                                       new int[5]{1, 1, 1, 1, 1});
    Layer *maxpool2 = new MaxThreeDPooling("maxpool2", new int[5]{1, 2, 2, 2, 1},
                                           new int[5]{1, 2, 2, 2, 1});

    Layer *conv3a = new ThreeDConvLayer("conv3a", 3, 3, 3, 128, 256, new int[5]{1, 1, 1, 1, 1},
                                        new int[5]{1, 1, 1, 1, 1});
    Layer *conv3b = new ThreeDConvLayer("conv3b", 3, 3, 3, 256, 256, new int[5]{1, 1, 1, 1, 1},
                                        new int[5]{1, 1, 1, 1, 1});
    Layer *maxpool3 = new MaxThreeDPooling("maxpool3", new int[5]{1, 2, 2, 2, 1},
                                           new int[5]{1, 2, 2, 2, 1});


    Layer *conv4a = new ThreeDConvLayer("conv4a", 3, 3, 3, 256, 512, new int[5]{1, 1, 1, 1, 1},
                                        new int[5]{1, 1, 1, 1, 1});
    Layer *conv4b = new ThreeDConvLayer("conv4b", 3, 3, 3, 512, 512, new int[5]{1, 1, 1, 1, 1},
                                        new int[5]{1, 1, 1, 1, 1});
    Layer *maxpool4 = new MaxThreeDPooling("maxpool4", new int[5]{1, 2, 2, 2, 1},
                                           new int[5]{1, 2, 2, 2, 1});

    Layer *conv5a = new ThreeDConvLayer("conv5a", 3, 3, 3, 512, 512, new int[5]{1, 1, 1, 1, 1},
                                        new int[5]{1, 1, 1, 1, 1});
    Layer *conv5b = new ThreeDConvLayer("conv5b", 3, 3, 3, 512, 512, new int[5]{1, 1, 1, 1, 1},
                                        new int[5]{1, 1, 1, 1, 1});
    Layer *maxpool5 = new MaxThreeDPooling("maxpool5", new int[5]{1, 2, 2, 2, 1},
                                           new int[5]{1, 2, 2, 2, 1});


    networkObj->layers.push_back(conv1);
    networkObj->layers.push_back(maxpool1);
    networkObj->layers.push_back(conv2);
    networkObj->layers.push_back(maxpool2);
    networkObj->layers.push_back(conv3a);
    networkObj->layers.push_back(conv3b);
    networkObj->layers.push_back(maxpool3);
    networkObj->layers.push_back(conv4a);
    networkObj->layers.push_back(conv4b);
    networkObj->layers.push_back(maxpool4);
    networkObj->layers.push_back(conv5a);
    networkObj->layers.push_back(conv5b);
    networkObj->layers.push_back(maxpool5);


    return reinterpret_cast<jlong>(networkObj);
}

extern "C"
JNIEXPORT void JNICALL
Java_edu_northeastern_nliu_c3d_NativeEntryInterface_loadParameters(JNIEnv *env, jobject instance,
                                                                   jlong networkPtr,
                                                                   jstring environmentPath_) {
    string path;
    const char *environmentPath = env->GetStringUTFChars(environmentPath_, 0);
    path = environmentPath;
    network *netObj = reinterpret_cast<network *>(networkPtr);

    for (int i = 0; i < netObj->layers.size(); i++) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Loading weights for Layer %s ",
                            netObj->layers[i]->_name.c_str());
        if (netObj->layers[i]->_layerType == LayerType::_3DCONVOLUTION) {

            ThreeDConvLayer *convLayer = dynamic_cast<ThreeDConvLayer *>(netObj->layers[i]);
            //load weights
            string weightspath;
            weightspath = path + convLayer->_name + "_weight";
            ifstream file(weightspath.c_str());
            int row = convLayer->_inCh * convLayer->_D * convLayer->_H * convLayer->_W;
            int col = convLayer->_outCh;
            string paraString;

            int count = 0;
            Eigen::Matrix<float, Dynamic, Dynamic> temp;
            temp.resize(row, col);
            while (getline(file, paraString)) {
                temp(count / col, count % col) = strtof(paraString.c_str(), 0);
                count++;
            }
            convLayer->_weightdense = temp;
            //load bias
            string biaspath;
            biaspath = path + convLayer->_name + "_bias";
            ifstream biasfile(biaspath.c_str());
            count = 0;
            Eigen::RowVectorXf bias(col);

            while (getline(biasfile, paraString)) {
                bias(count) = strtof(paraString.c_str(), 0);
                count++;
            }
            convLayer->_bias = bias;
//            __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,"%s",to_string(convLayer->_weightdense(0,1)).c_str());
        } else if (netObj->layers[i]->_layerType == LayerType::_3DPOOL) {
            continue;
        }
    }

    env->ReleaseStringUTFChars(environmentPath_, environmentPath);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_edu_northeastern_nliu_c3d_NativeEntryInterface_classify(JNIEnv *env, jobject instance,
                                                             jlong networkPtr,
                                                             jfloatArray img_,
                                                             jintArray imgShape_) {
    jfloat *img = env->GetFloatArrayElements(img_, NULL);
    jint *imgShape = env->GetIntArrayElements(imgShape_, NULL);
    network *net = reinterpret_cast<network *>(networkPtr);
    // TODO
    Eigen::Matrix<float, Dynamic, Dynamic> imageMat;
    int row = imgShape[0] * imgShape[1] * imgShape[2];
    int col = imgShape[3];
    imageMat.resize(row, col);
    for (int i = 0; i < row * col; i++) {
        imageMat(i / col, i % col) = img[i];
    }

    env->ReleaseFloatArrayElements(img_, img, 0);
    env->ReleaseIntArrayElements(imgShape_, imgShape, 0);

    string result;
    net->_inShape = imgShape;
    clock_t start;
    double eclipse;
    start = clock();
    result = net->predict(imageMat);
    eclipse = ((clock() - start)) / (double) CLOCKS_PER_SEC;
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "classify time %s ",
                        to_string(eclipse).c_str());
    return env->NewStringUTF((result).c_str());
}
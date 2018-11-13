//
// Created by nliu on 9/25/18.
//

#include <android/log.h>
#include "network.h"

using namespace std;
using namespace Eigen;

#define  LOG_TAG    "debug::network.cpp"

//************************************************************************
Layer::Layer(string name) {
    _name = name;
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Layer %s constructed", _name.c_str());
}

void Layer::forward() {

}

//************************************************************************
ThreeDConvLayer::ThreeDConvLayer(string name, int D, int H, int W, int inCh, int outCh,
                                 int *strides, int *padding) : Layer(name) {
    _D = D;
    _H = H;
    _W = W;
    _inCh = inCh;
    _outCh = outCh;
    _strides = strides;
    _padding = padding;
    _layerType = LayerType::_3DCONVOLUTION;
}

void ThreeDConvLayer::forward() {
    int outputCol = this->_outCh;
    int outputRow = ((_inputShape[0] + _padding[0] * 2 - _D) / _strides[0] + 1)
                    * ((_inputShape[1] + _padding[1] * 2 - _H) / _strides[1] + 1)
                    * ((_inputShape[2] + _padding[2] * 2 - _W) / _strides[2] + 1);
    int colBufferCol = _H * _W * _inCh * _D;
    _colBuffer.resize(outputRow, colBufferCol);
    im2col();
    Eigen::Matrix<float, Dynamic, Dynamic> temp;

    //multiplication for convlution
    temp = _colBuffer * _weightdense;

//    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "layer %s value of certain point is %s",
//                        _name.c_str(), to_string(temp(11, 11)).c_str());
    //add bias

    temp.rowwise() += _bias;

//    temp.array() = temp.array().cwiseMax(0);//relu
    temp = temp.cwiseMax(0);//relu

    _output = temp;
    _outputShape = new int[4]{((_inputShape[0] + _padding[0] * 2 - _D) / _strides[0] + 1),
                              ((_inputShape[1] + _padding[1] * 2 - _H) / _strides[1] + 1),
                              ((_inputShape[2] + _padding[2] * 2 - _W) / _strides[2] + 1),
                              outputCol};

}

//
void ThreeDConvLayer::im2col() {
    //move convolution window
    int row = 0;
    //padded input
    for (int Din = 0; Din < (_inputShape[0] + _padding[0] * 2 - _D) + 1; Din = Din + _strides[0]) {
        for (int Hin = 0;
             Hin < (_inputShape[1] + _padding[1] * 2 - _H) + 1; Hin = Hin + _strides[1]) {
            for (int Win = 0;
                 Win < (_inputShape[2] + _padding[2] * 2 - _W) + 1; Win = Win + _strides[2]) {
                //inside for the filter kernel
                int column = 0;
                for (int Dw = Din; Dw < Din + _D; Dw++) {
                    for (int Hw = Hin; Hw < Hin + _H; Hw++) {
                        for (int Ww = Win; Ww < Win + _W; Ww++) {
                            for (int InChw = 0; InChw < _inCh; InChw++) {
                                float value = getData(Dw, Hw, Ww, InChw);
                                _colBuffer(row, column) = value;
                                column++;
                            }
                        }
                    }
                }
                row++;
            }
        }
    }
}


float ThreeDConvLayer::getData(int Din, int Hin, int Win, int InChin) {
    if (Din < _padding[0] || Din >= _inputShape[0] + _padding[0]
        || Hin < _padding[1] || Hin >= _inputShape[1] + _padding[1]
        || Win < _padding[2] || Win >= _inputShape[2] + _padding[2]
            ) {
        return 0;
    } else {
        int offsetD = _inputShape[1] * _inputShape[2];
        int offsetH = _inputShape[2];
        return _input((Din - _padding[0]) * offsetD
                      + ((Hin - _padding[1]) * offsetH)
                      + ((Win - _padding[2])), InChin);
    }
}

//************************************************************************
network::network(string name) {
    _name = name;
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "network %s constructed", _name.c_str());

}
//************************************************************************

float* network::predict(Eigen::Matrix<float, Dynamic, Dynamic> image) {
    Eigen::Matrix<float, Dynamic, Dynamic> finaloutput;
    for (int i = 0; i < layers.size(); i++) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Layer %s proccessing",
                            layers[i]->_name.c_str());
        if (i == 0) {
            layers[i]->_input = image;
            layers[i]->_inputShape = this->_inShape;
        } else {
            layers[i]->_input = layers[i - 1]->_output;
            layers[i]->_inputShape = layers[i - 1]->_outputShape;

        }
        layers[i]->forward();
        if (i == layers.size() - 1) {
            finaloutput = layers[i]->_output;
//            __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "last output certain point value is %s",
//                                to_string(finaloutput(0, 0)).c_str());
        }
    }

    return finaloutput.data();
}


MaxThreeDPooling::MaxThreeDPooling(string name, int *ksize, int *strides) : Layer(name) {
    _ksize = ksize;
    _strides = strides;
    _name = name;
    _layerType = LayerType::_3DPOOL;
}

void MaxThreeDPooling::forward() {
    //calculate padding
    calPadding();
    int col = _input.cols();
    int outD = (_inputShape[0] + _padding[1] - _ksize[1]) / _strides[1] + 1;
    int outH = (_inputShape[1] + _padding[2] - _ksize[2]) / _strides[2] + 1;
    int outW = (_inputShape[2] + _padding[3] - _ksize[3]) / _strides[3] + 1;
    _output.resize(outD * outH * outW, col);
    _outputShape = new int[4]{outD, outH, outW, col};
    //start indexing


    for (int CHin = 0; CHin < col; ++CHin) {
        int row = 0;
        for (int Din = 0;
             Din < (_inputShape[0] + _padding[1] - _ksize[1]) + 1; Din = Din + _strides[1]) {
            for (int Hin = 0;
                 Hin < (_inputShape[1] + _padding[2] - _ksize[2]) + 1; Hin = Hin + _strides[2]) {
                for (int Win = 0; Win < (_inputShape[2] + _padding[3] - _ksize[3]) + 1; Win = Win +
                                                                                              _strides[3]) {
                    //inside for the filter kernel
                    float value = FLT_MIN;
                    for (int Dw = Din; Dw < Din + _ksize[1]; Dw++) {
                        for (int Hw = Hin; Hw < Hin + _ksize[2]; Hw++) {
                            for (int Ww = Win; Ww < Win + _ksize[3]; Ww++) {
                                value = max(getData(Dw, Hw, Ww, CHin), value);
                            }
                        }
                    }
                    _output(row, CHin) = value;
                    row++;
                }
            }
        }
    }
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "layer %s value of certain point is %s",
                        _name.c_str(),to_string(_output(11, 11)).c_str());
}

float MaxThreeDPooling::getData(int Din, int Hin, int Win, int InChin) {
    if (Din > _inputShape[0] - 1
        || Hin > _inputShape[1] - 1
        || Win > _inputShape[2] - 1) {
        return 0;
    } else {
        int offsetD = _inputShape[1] * _inputShape[2];
        int offsetH = _inputShape[2];
        return _input(Din * offsetD + Hin * offsetH + Win, InChin);
    }
}

void MaxThreeDPooling::calPadding() {
    //inShape 4d DHWC_in,ksize 5d,strides 5d, to calculate paddings 5d
    int *pad = new int[5]{};
    pad[0] = 0;
    for (int i = 0; i < 5; ++i) {
        if (i > 0) {
            pad[i] = _inputShape[i - 1] % _strides[i];
        }
    }
    _padding = pad;
}

//************************************************************************
FlattenLayer::FlattenLayer(string name, int Row, int Col) : Layer(name) {
    _Row = Row;
    _Col = Col;
    _name = name;
    _layerType = LayerType::_FLATTEN;
}

void FlattenLayer::forward() {
    int flattenSize = _Row * _Col;
    _input.resize(1, flattenSize);
    _output = _input;

}


//************************************************************************
FCLayer::FCLayer(string name, int In, int Out, string activate) : Layer(name) {
    _activate = activate;
    _In = In;
    _Out = Out;
    _name = name;
    _layerType = LayerType::_FC;
}

void FCLayer::forward() {
    Eigen::Matrix<float, Dynamic, Dynamic> temp;
    temp = _input * _weightdense;
    temp.rowwise() += _bias;

    if (_activate=="relu") {
        temp = temp.cwiseMax(0); //relu
    } else if (_activate=="softmax"){
      //not implemented
      temp=temp;
    }
    _output = temp;

}
//************************************************************************
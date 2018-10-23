//
// Created by nliu on 9/25/18.
//

#ifndef C3D_NETWORK_H
#define C3D_NETWORK_H


#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;


enum class LayerType {
    _3DCONVOLUTION,
    _3DPOOL
    };

class Layer{
public:
    Layer(string name);

    Layer();

    virtual ~Layer(){};
/*****************************************************/
    string _name;
    LayerType _layerType;
    Eigen::RowVectorXf _bias;
//    Eigen::Matrix <float, Dynamic,Dynamic> _bias;
    Eigen::Matrix <float, Dynamic,Dynamic> _weightdense;
    Eigen::Matrix <float, Dynamic,Dynamic> _input;
    int* _inputShape;
    Eigen::Matrix <float, Dynamic,Dynamic> _output;
    Eigen::Matrix <float, Dynamic,Dynamic> _colBuffer;
    int* _outputShape;
/****************************************************/
    virtual void forward();
    void activation(Eigen::Matrix <float, Dynamic,Dynamic> &matrix);
};



class ThreeDConvLayer:public Layer{
public:
    ThreeDConvLayer(string name, int D, int H, int W, int inCh, int outCh, int *strides,
                        int *padding);

    ~ThreeDConvLayer(){};
    int _D;
    int _H;
    int _W;
    int _inCh;
    int _outCh;
    int* _strides;
    int* _padding;
    //fuctions
    void forward();
    void im2col();
    float getData(int Din,int Hin,int Win,int InChin);
};

class MaxThreeDPooling:public Layer{
public:
    MaxThreeDPooling(string name, int *ksize, int *strides);
    ~MaxThreeDPooling(){};
    int* _ksize;
    int* _strides;
    int* _padding;
    //fuctions
    void calPadding();
    void forward();
    float getData(int Din,int Hin,int Win,int InChin);
};

class network {
public:
    network(string name);
    ~network() {
        cout<< "network deconstrcuted"<<endl;

    }
/****************************************************/
    //input shape
    vector<Layer*> layers;
    string _name;
    int* _inShape;
    Eigen::Matrix <float, Dynamic,Dynamic> _input;
/****************************************************/
    string predict(Eigen::Matrix <float, Dynamic,Dynamic> image);

};


#endif //C3D_NETWORK_H

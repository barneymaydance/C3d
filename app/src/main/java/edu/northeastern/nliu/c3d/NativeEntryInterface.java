package edu.northeastern.nliu.c3d;

import android.util.Log;

public class NativeEntryInterface {


    public native long buildNetwork();
    public native void loadParameters(long networkPtr, String environmentPath);
    public native String classify(long networkPtr, float[] img, int[] imgShape);


}

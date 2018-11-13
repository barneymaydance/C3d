package edu.northeastern.nliu.sst;

import android.util.Log;

public class NativeEntryInterface {


    public native long buildC3DNetwork();
    public native void loadC3DParameters(long networkPtr, String environmentPath);
    public native float[] classify(long networkPtr, float[] img, int[] imgShape);


}

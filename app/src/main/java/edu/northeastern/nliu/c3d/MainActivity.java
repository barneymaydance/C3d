package edu.northeastern.nliu.c3d;

import android.graphics.Bitmap;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();

    private static final int CROP_WIDTH=112;
    private static final int NB_PER_FRAME=16;
    private static final int RGB_CHANNEL=3;

    static {
        System.loadLibrary("native-lib");
    }

    private Button bt_ldweight;
    private Button bt_classify;
    private TextView tv_ldweights;
    private TextView tv_classify;
    private NativeEntryInterface client;
    private long networkPtr;
    private int[] imgShape;
    private static String environmentPath;
    private List imageCollection;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        bt_ldweight = (Button) findViewById(R.id.button_lw);
        bt_classify = (Button) findViewById(R.id.button_classify);
        tv_ldweights = (TextView) findViewById(R.id.tv_loadweights);
        tv_classify = (TextView) findViewById(R.id.tv_classify);
        client= new NativeEntryInterface();

        //set img path
        environmentPath= Environment.getExternalStorageDirectory().getAbsolutePath() + "/c3d_resource/";
        Log.d(TAG, environmentPath);

        //build network
        //DHWC
        networkPtr = client.buildNetwork();

        //load weights
        bt_ldweight.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                // Code here executes on main thread after user presses button
                client.loadParameters(networkPtr, environmentPath);
                tv_ldweights.setText("weight loaded!");

            }
        });
        //read img and classify
        imgShape = new int[]{NB_PER_FRAME, CROP_WIDTH, CROP_WIDTH, RGB_CHANNEL};
        bt_classify.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {

                float[] img = loadImg(environmentPath + "input");
                long startTime = System.currentTimeMillis();
                client.classify(networkPtr, img, imgShape);
                long endTime = System.currentTimeMillis();
                long runTime = endTime - startTime;
                tv_classify.setText(String.format("%d", runTime));

            }
        });

    }

    private float[] loadImg(String path) {
        float[] image = new float[NB_PER_FRAME * CROP_WIDTH * CROP_WIDTH * RGB_CHANNEL];
        File file = new File(path);

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            int count = 0;
            while ((line = br.readLine()) != null) {
                image[count] = Float.valueOf(line);
                count++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }

    private static ArrayList<Bitmap> loadImgFromStorage(String path){

        File file = new File(path);
        return new ArrayList();

    }
}
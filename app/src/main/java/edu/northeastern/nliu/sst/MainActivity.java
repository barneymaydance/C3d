package edu.northeastern.nliu.sst;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();

    private static final int CROP_WIDTH = 112;
    private static final int NB_PER_FRAME = 16;
    private static final int RGB_CHANNEL = 3;

    static {
        System.loadLibrary("native-lib");
    }

//    private static final String[] activityList={"抽烟","玩手机","跳高","两人拉扯","挥手","点头","摇头"};
    private static final String[] activityList={"跳高","挥手","摇头"};
//    private static final String[] activityList = {"BasketballDunk", "CleanAndJerk", "CliffDiving", "FrisbeeCatch", "HammerThrow",
//            "HighJump", "JavelinThrow","LongJump","PoleVault","SoccerPenalty"};//ucf101 dataset select 10 classes

    private Button bt_ldweight;
    private Button bt_classify;
    private Button bt_clear;
    private TextView tv_ldweights;
    private TextView tv_classify;
    private NativeEntryInterface client;
    private long networkPtr;
    private int[] imgShape;
    private String environmentPath;

    private List imageCollection;

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case 1: {

                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.
                } else {

                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                    Toast.makeText(MainActivity.this, "Permission denied to read your External storage", Toast.LENGTH_SHORT).show();
                }
                return;
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ActivityCompat.requestPermissions(MainActivity.this,
                new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                1);

        bt_ldweight = (Button) findViewById(R.id.button_lw);
        bt_classify = (Button) findViewById(R.id.button_classify);
        bt_clear = (Button) findViewById(R.id.button_clear);

        tv_ldweights = (TextView) findViewById(R.id.tv_loadweights);
        tv_classify = (TextView) findViewById(R.id.tv_classify);
        client = new NativeEntryInterface();

        //set img path
        environmentPath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/c3d_resource/";

        Log.d(TAG, environmentPath);

        //build network
        //DHWC
        networkPtr = client.buildC3DNetwork();

        //load weights
        bt_ldweight.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                // Code here executes on main thread after user presses button
                client.loadC3DParameters(networkPtr, environmentPath);
                tv_ldweights.setText("Weight loaded!");

            }
        });
        //read img and classify
        imgShape = new int[]{NB_PER_FRAME, CROP_WIDTH, CROP_WIDTH, RGB_CHANNEL};
        bt_classify.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {

//                loadImgFromStorage(environmentPath+"predict_score.bin");
                float[] img = loadImg(environmentPath + "input");
                float[] logits=new float[1];
                long startTime;
                long endTime;
                long runTime=0;
                for (int i=0;i<10;i++){
                    startTime = System.currentTimeMillis();
                    logits= client.classify(networkPtr, img, imgShape);
                    endTime = System.currentTimeMillis();
                    runTime = endTime - startTime;
                    Log.v(TAG, String.format("Repeat: %d c3d, time %d", i, runTime));
                }

                int predictIndx = 0;
                float curMax = Float.MAX_VALUE;
                for (int i = 1; i <logits.length;  i++) {
                    if (logits[i] > curMax) {
                        curMax = logits[i];
                        predictIndx = i;
                    }
                }
                String out = "";
                for (float n : logits) {
                    out = out + Float.toString(n);
                }

                tv_classify.setText(String.format("Time: %d ms, %d", runTime, predictIndx));

            }
        });

        bt_clear.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                tv_ldweights.setText("");
                tv_classify.setText("");
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

    private List loadImgFromStorage(String path) {
//        imageCollection=new ArrayList<Bitmap>();
        imageCollection = new ArrayList<Float>();
        try {
            FileInputStream fin = new FileInputStream(path);
            BufferedInputStream bin = new BufferedInputStream(fin);
            DataInputStream din = new DataInputStream(bin);

            float floatRead;

            while (true) {
                floatRead = din.readFloat();
                imageCollection.add(floatRead);
            }

        } catch (IOException ex) {
            ex.printStackTrace();
        }

        return imageCollection;

    }


}
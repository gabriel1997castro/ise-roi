package com.example.roi;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;

import android.util.Log;
import android.view.SurfaceView;
import android.widget.Toast;


import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {
    private CameraBridgeViewBase mOpenCvCameraView;
    private static final String TAG = "OCVSample::Activity";

    private Mat mat1;

    BaseLoaderCallback baseLoaderCallback;
    private CascadeClassifier cascadeClassifier;
    private int absoluteFaceSize;

//    BaseLoaderCallback mCallBack = new BaseLoaderCallback(this) {
//        @Override
//        public void onManagerConnected(int status) {
//            if (status == BaseLoaderCallback.SUCCESS) {
//                Log.i(TAG, "OpenCV loaded successfully");
//                mOpenCvCameraView.enableView();
//            } else {
//                super.onManagerConnected(status);
//            }
//        }
//    };


    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, baseLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.MyView);
        mOpenCvCameraView.setCameraIndex(1);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);


        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {

                switch (status) {
                    case BaseLoaderCallback.SUCCESS:
                        initializeOpenCVDependencies();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;

                }
            }
        };
    }


    private void initializeOpenCVDependencies() {

        try {
            // Copy the resource into a temp file so OpenCV can load it
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);


            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Load the cascade classifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());

        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }

        // And we are ready to go
        mOpenCvCameraView.enableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mat1 = new Mat(width, height, CvType.CV_8UC4);

    }

    @Override
    public void onCameraViewStopped() {
        mat1.release();

    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mat1 = inputFrame.rgba();
        Mat mRgbaT = mat1.t();
        // Flip the image of the frontal camera, because it starts upside down
        Core.flip(mat1.t(), mRgbaT, -1);
        Imgproc.resize(mRgbaT, mRgbaT, mat1.size());

        Imgproc.cvtColor(mRgbaT, mRgbaT, Imgproc.COLOR_RGBA2RGB);

        MatOfRect faces = new MatOfRect();

        // Use the classifier to detect faces
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(mRgbaT, faces, 1.1, 2, 2,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }

        Rect roiRect;
        Mat roiMat;
        Rect[] facesArray = faces.toArray();
        if(facesArray.length > 0) {
            Rect face = getNearestFace(facesArray);

            Point br = face.br();
            Point tl = face.tl();

            int SQUARE_SIZE = (int) (face.height*0.3);
            roiRect = new Rect((int) face.tl().x+face.width-SQUARE_SIZE/2, (int) face.tl().y, SQUARE_SIZE, SQUARE_SIZE);
            roiMat = new Mat(mRgbaT, roiRect);
            Log.i(TAG, "ROI detected ------> roiMat: " + roiMat + " RoiRect: " + roiRect);

            // get only 30% from top to bottom - only forehead
            br.y = tl.y + face.height*0.3;
            Imgproc.rectangle(mRgbaT, tl, br, new Scalar(0, 255, 0, 255), 3);
        }

        return mRgbaT;


    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }

    }

    private Rect getNearestFace(Rect[] facesArray) {
        Rect biggerFace = facesArray[0];
//        for(int i = 0; i < facesArray.length; i++) {
//            if(i > 0 && (facesArray[i].width * facesArray[i].height > facesArray[i-1].width *facesArray[i-1].height)) {
//                biggerFace = facesArray[i];
//            }
//        }
        return biggerFace;
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(getApplicationContext(), "There is problem in OpenCV", Toast.LENGTH_SHORT).show();
        }
    }


}
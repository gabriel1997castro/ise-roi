/*
* 1. Get camera frama
* 2. Get faces from camera
* 3. Select one face
* 5. Get forehead and draw rectangle on it
* 6. Get roi squared
* 7. Separate colors
* 8. Apply DCT
* 9. Apply Zig-Zag transform
* 10. Calculate energy
* 11. Calculate saturation
* */

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
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {
    private CameraBridgeViewBase mOpenCvCameraView;
    private static final String TAG = "OCVSample::Activity";

    private Mat mat1, roi1, roi2;

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


    static void zigZag(Mat arr, int n, int m, ArrayList<Integer> result) {
        int row = 0, col = 0;
        double[] temp = new double[1];
        boolean row_inc = false;

        int mn = Math.min(m, n);
        for (int len = 1; len <= mn; ++len) {
            for (int i = 0; i < len; ++i) {
                arr.get(row,col, temp);
                result.add((int)temp[0]);

                if (i + 1 == len)
                    break;
                if (row_inc) {
                    ++row;
                    --col;
                } else {
                    --row;
                    ++col;
                }
            }

            if (len == mn)
                break;
            if (row_inc) {
                ++row;
                row_inc = false;
            } else {
                ++col;
                row_inc = true;
            }
        }

        if (row == 0) {
            if (col == m - 1)
                ++row;
            else
                ++col;
            row_inc = true;
        } else {
            if (row == n - 1)
                ++col;
            else
                ++row;
            row_inc = false;
        }

        int MAX = Math.max(m, n) - 1;
        for (int len, diag = MAX; diag > 0; --diag) {

            if (diag > mn)
                len = mn;
            else
                len = diag;

            for (int i = 0; i < len; ++i) {
                arr.get(row,col, temp);
                result.add((int)temp[0]);

                if (i + 1 == len)
                    break;

                if (row_inc) {
                    ++row;
                    --col;
                } else {
                    ++col;
                    --row;
                }
            }
            if (row == 0 || col == m - 1) {
                if (col == m - 1)
                    ++row;
                else
                    ++col;

                row_inc = true;
            } else if (col == 0 || row == n - 1) {
                if (row == n - 1)
                    ++col;
                else
                    ++row;

                row_inc = false;
            }
        }
    }

    private double calculateAverage(List <Integer> list) {
        Double sum = 0.0;
        if(!list.isEmpty()) {
            for (Integer item : list) {
                sum += item * item;
            }
            return sum.doubleValue() / list.size();
        }
        return sum;
    }

//    static float calculateEnergy(ArrayList<Integer> result) {
//        double percentageOfRemove = 0.7;
//        int size = (int)(result.size()*percentageOfRemove);
//        if (size == 0) {
//            return 0;
//        }
//        int bound = Math.round(size/12);
//        float majorPeak = 0;
//        float energy = 0;
//        for(int i=0; i < size; i = i+bound) {
//            majorPeak = Collections.max(result.subList(i, i+bound));
//            energy += majorPeak*majorPeak;
//        }
//        return energy;
//    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat mRgbaT = mat1.t();
        try {
            mat1 = inputFrame.rgba();

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
            // -------------------------------------------------------------------------------------- //

            Rect roiRect;
            Mat roiMat = null;
            Rect[] facesArray = faces.toArray();
            if(facesArray.length > 0) {
                Rect face = getNearestFace(facesArray);

                Point br = face.br();
                Point tl = face.tl();

                int SQUARE_SIZE = (int) (face.height*0.3);
                int MARGIN_SIZE = (int) (face.height*0.05);
                int startPoint = (int) face.tl().x+face.width/2-SQUARE_SIZE/2 + MARGIN_SIZE;
                int endPoint = (int) face.tl().y + MARGIN_SIZE;
                int SQUARE_SIZE_WITH_MARGIN = SQUARE_SIZE-2*MARGIN_SIZE % 2 ==1 ? SQUARE_SIZE-2*MARGIN_SIZE - 1 : SQUARE_SIZE-2*MARGIN_SIZE;
                roiRect = new Rect(startPoint, endPoint, 80, 80);
                roiMat = new Mat(mRgbaT, roiRect);
//                Log.i(TAG, "ROI detected ------> roiMat: " + roiMat + " RoiRect: " + roiRect);

                // get only 30% from top to bottom - only forehead
                br.y = tl.y + face.height*0.3;
                // Draw on forehead
                Imgproc.rectangle(mRgbaT, tl, br, new Scalar(0, 255, 0, 255), 3);
                // Draw the roi square
                Imgproc.rectangle(mRgbaT, roiRect.tl(), roiRect.br(), new Scalar(0, 0, 255, 255), 3);

                if(SQUARE_SIZE_WITH_MARGIN < 80) {
                    String text = "aproxime";
                    Imgproc.putText(mRgbaT, text, new Point(mRgbaT.cols() / 2 - 200, mRgbaT.rows() - 40), Core.FONT_ITALIC, 2.0, new Scalar(255, 255, 255), 2);
                    return mRgbaT;
                }
                if(SQUARE_SIZE_WITH_MARGIN > 100) {
                    String text = "afaste";
                    Imgproc.putText(mRgbaT, text, new Point(mRgbaT.cols() / 2 - 200, mRgbaT.rows() - 40), Core.FONT_ITALIC, 2.0, new Scalar(255, 255, 255), 2);
                    return mRgbaT;
                }
            }
            // -------------------------------------------------------------------------------------- //

            roi2 = roi1;
            roi1 = roiMat;



            if(roiMat != null && roi1 != null && roi2 != null) {

                List<Mat> bgr = new ArrayList();

                Mat roi = new Mat();
                Core.subtract(roi2, roi1, roi);
                Core.split(roi, bgr);

                Mat blue = bgr.get(0);
                Mat red = bgr.get(2);

                blue.convertTo(blue, CvType.CV_32FC1);
                red.convertTo(red,CvType.CV_32FC1);
                Core.dct(blue, blue);
                Core.dct(red, red);
                blue.convertTo(blue,CvType.CV_64FC1);
                red.convertTo(red,CvType.CV_64FC1);

                ArrayList<Integer> zzBlue = new ArrayList<Integer>();
                ArrayList<Integer> zzRed = new ArrayList<Integer>();

                zigZag(blue, blue.rows(), blue.cols(), zzBlue);
                zigZag(red, red.rows(), red.cols(), zzRed);

                double redMax = Collections.max(zzRed);
                double blueMax = Collections.max(zzBlue);
                System.out.println("hello");
                double redMean = calculateAverage(zzRed);
                double blueMean = calculateAverage(zzBlue);

                double r = (redMax / redMean) / (blueMax / blueMean);

                double k = 10.5;
                double spo2 = 100.0 - r * k;

                String text = String.format("SPO2: %.2f %%", spo2);
                Imgproc.putText(mRgbaT,text,new Point(mRgbaT.cols()/2-300,mRgbaT.rows()-40),Core.FONT_ITALIC,2.0,new Scalar(255,255,255),2);
//
            }

        } catch (Exception e) {
            System.out.println(e.getMessage());
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
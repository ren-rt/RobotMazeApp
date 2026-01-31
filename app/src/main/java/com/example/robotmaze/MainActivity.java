package com.example.robotmaze;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 100;

    static {
        if (OpenCVLoader.initLocal()) {
            Log.d(TAG, "OpenCV loaded successfully!");
        } else {
            Log.e(TAG, "OpenCV initialization failed.");
        }
    }

    private PreviewView previewView;
    private ImageView processedImageView;
    private Button captureButton;
    private Button backButton;
    private ScrollView gridScrollView;
    private TextView gridTextView;
    private ImageCapture imageCapture;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        previewView = findViewById(R.id.preview_view);
        processedImageView = findViewById(R.id.processed_image_view);
        captureButton = findViewById(R.id.capture_button);
        backButton = findViewById(R.id.back_button);
        gridScrollView = findViewById(R.id.grid_scroll_view);
        gridTextView = findViewById(R.id.grid_text_view);

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(
                    this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        }

        backButton.setOnClickListener(v -> showPreview());
    }

    public void onCaptureButtonClick(View view) {
        imageCapture.takePicture(ContextCompat.getMainExecutor(this), new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                Bitmap bitmap = imageProxyToBitmap(image);
                Bitmap processedBitmap = processMaze(bitmap);
                showProcessedImage(processedBitmap);
                image.close();
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e("CameraX", "Capture failed: " + exception.getMessage());
            }
        });
    }

    private Bitmap processMaze(Bitmap bitmap) {
        Mat originalMat = new Mat();
        Utils.bitmapToMat(bitmap, originalMat);

        Mat grayTemp = new Mat();
        Imgproc.cvtColor(originalMat, grayTemp, Imgproc.COLOR_RGBA2GRAY);
        Mat binaryTemp = new Mat();
        Imgproc.threshold(grayTemp, binaryTemp, 128, 255, Imgproc.THRESH_BINARY);
        Point[] corners = findMazeCorners(binaryTemp);

        Mat processedMat;
        if (corners != null) {
            Mat warpedMat = warpMaze(originalMat, corners);
            Imgproc.cvtColor(warpedMat, warpedMat, Imgproc.COLOR_RGBA2BGR);
            processedMat = robustClean(warpedMat);
            warpedMat.release();
        } else {
            Imgproc.cvtColor(originalMat, originalMat, Imgproc.COLOR_RGBA2BGR);
            processedMat = robustClean(originalMat);
        }

        int[][] grid = convertToLogicGrid(processedMat, 100);
        displayGrid(grid);

        Mat displayMat = debugVisualGrid(grid, 10, 10);

        Bitmap processedBitmap = Bitmap.createBitmap(displayMat.cols(), displayMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(displayMat, processedBitmap);

        // Release memory
        originalMat.release();
        grayTemp.release();
        binaryTemp.release();
        processedMat.release();
        displayMat.release();

        return processedBitmap;
    }

    private Mat robustClean(Mat warpedMat) {
        Mat gray = new Mat();
        Mat binary = new Mat();

        // 1. Gray and Blur
        Imgproc.cvtColor(warpedMat, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

        // 2. Contrast Stretch (The Fix for "All White" images)
        // This forces the maze walls to be dark even in bright light
        Core.normalize(gray, gray, 0, 255, Core.NORM_MINMAX);

        // 3. OTSU Thresholding (The "Smart" Threshold)
        // Instead of choosing a number like 128, OTSU analyzes the
        // histogram to find the "perfect" split point between path and wall.
        Imgproc.threshold(gray, binary, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        // 4. De-Noising
        // This removes those "random black blocks" that aren't walls
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernel);

        return binary;
    }

    private Mat debugVisualGrid(int[][] grid, int rows, int cols) {
        Mat debugMat = new Mat(rows, cols, CvType.CV_8UC1);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                // Walls = Black(0), Paths = White(255)
                double val = (grid[r][c] == 1) ? 0 : 255;
                debugMat.put(r, c, val);
            }
        }
        // Zoom it back up to see it clearly
        Mat displayMat = new Mat();
        Imgproc.resize(debugMat, displayMat, new Size(1000, 1000), 0, 0, Imgproc.INTER_NEAREST);
        debugMat.release();
        return displayMat;
    }

    public int[][] convertToLogicGrid(Mat binaryMat, int gridSize) {
        int rows = binaryMat.rows() / gridSize;
        int cols = binaryMat.cols() / gridSize;
        int[][] logicGrid = new int[rows][cols];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {

                boolean isWall = false;
                // Check a 5x5 area around the center of the grid cell
                for (int i = -2; i <= 2; i++) {
                    for (int j = -2; j <= 2; j++) {
                        int checkR = r * gridSize + (gridSize / 2) + i;
                        int checkC = c * gridSize + (gridSize / 2) + j;

                        // Stay within bounds
                        if (checkR >= 0 && checkR < binaryMat.rows() && checkC >= 0 && checkC < binaryMat.cols()) {
                            double[] pixel = binaryMat.get(checkR, checkC);
                            if (pixel[0] < 120) { // If ANY pixel is dark enough
                                isWall = true;
                                break;
                            }
                        }
                    }
                    if (isWall) break;
                }
                logicGrid[r][c] = isWall ? 1 : 0;
            }
        }
        return logicGrid;
    }

    private void displayGrid(int[][] grid) {
        StringBuilder gridText = new StringBuilder();
        for (int[] row : grid) {
            for (int cell : row) {
                gridText.append(cell).append(" ");
            }
            gridText.append("\n");
        }
        gridTextView.setText(gridText.toString());
    }

    private Mat warpMaze(Mat input, Point[] corners) {
        Point[] sortedCorners = sortCorners(corners);

        MatOfPoint2f src = new MatOfPoint2f(sortedCorners);
        MatOfPoint2f dst = new MatOfPoint2f(
                new Point(0, 0),
                new Point(1000, 0),
                new Point(1000, 1000),
                new Point(0, 1000)
        );

        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(src, dst);

        Mat warped = new Mat();
        Imgproc.warpPerspective(input, warped, perspectiveTransform, new Size(1000, 1000));
        perspectiveTransform.release();
        return warped;
    }

    private Point[] sortCorners(Point[] corners) {
        List<Point> cornerList = new ArrayList<>();
        for (Point corner : corners) {
            cornerList.add(corner);
        }
        Collections.sort(cornerList, (p1, p2) -> Double.compare(p1.y, p2.y));
        if (cornerList.get(0).x > cornerList.get(1).x) {
            Collections.swap(cornerList, 0, 1);
        }
        if (cornerList.get(2).x < cornerList.get(3).x) {
            Collections.swap(cornerList, 2, 3);
        }

        Point[] sortedCorners = new Point[4];
        cornerList.toArray(sortedCorners);
        return sortedCorners;
    }


    private Point[] findMazeCorners(Mat binary) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        MatOfPoint largestContour = null;
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                largestContour = contour;
            }
        }

        if (largestContour != null) {
            MatOfPoint2f contour2f = new MatOfPoint2f(largestContour.toArray());
            double peri = Imgproc.arcLength(contour2f, true);
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approx, 0.02 * peri, true);

            if (approx.total() == 4) {
                Point[] result = approx.toArray();
                approx.release();
                contour2f.release();
                largestContour.release();
                return result;
            }
        }
        return null;
    }


    private void showProcessedImage(Bitmap bitmap) {
        previewView.setVisibility(View.GONE);
        captureButton.setVisibility(View.GONE);
        processedImageView.setVisibility(View.VISIBLE);
        backButton.setVisibility(View.VISIBLE);
        gridScrollView.setVisibility(View.VISIBLE);
        processedImageView.setImageBitmap(bitmap);
    }

    private void showPreview() {
        previewView.setVisibility(View.VISIBLE);
        captureButton.setVisibility(View.VISIBLE);
        processedImageView.setVisibility(View.GONE);
        backButton.setVisibility(View.GONE);
        gridScrollView.setVisibility(View.GONE);
    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);

        Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);

        Matrix matrix = new Matrix();
        matrix.postRotate(image.getImageInfo().getRotationDegrees());

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                imageCapture = new ImageCapture.Builder()
                    .setTargetRotation(getWindowManager().getDefaultDisplay().getRotation())
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build();

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture);

            } catch (Exception e) {
                Log.e("CameraX", "Binding failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }
}
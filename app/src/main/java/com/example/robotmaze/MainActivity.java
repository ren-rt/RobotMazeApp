package com.example.robotmaze;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
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
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
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
    private View captureUploadLayout;
    private View processBackLayout;
    private ScrollView gridScrollView;
    private TextView gridTextView;
    private ImageCapture imageCapture;
    private Bitmap originalBitmap;
    private ActivityResultLauncher<Intent> pickImageLauncher;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        previewView = findViewById(R.id.preview_view);
        processedImageView = findViewById(R.id.processed_image_view);
        captureUploadLayout = findViewById(R.id.capture_upload_layout);
        processBackLayout = findViewById(R.id.process_back_layout);
        Button backButton = findViewById(R.id.back_button);
        gridScrollView = findViewById(R.id.grid_scroll_view);
        gridTextView = findViewById(R.id.grid_text_view);

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(
                    this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        }

        backButton.setOnClickListener(v -> showPreview());

        pickImageLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null && result.getData().getData() != null) {
                        Uri imageUri = result.getData().getData();
                        try (InputStream inputStream = getContentResolver().openInputStream(imageUri)) {
                            originalBitmap = BitmapFactory.decodeStream(inputStream);
                            displayBinarizedImage(originalBitmap);
                        } catch (IOException e) {
                            Log.e(TAG, "Failed to load image from gallery", e);
                        }
                    }
                });
    }

    public void onCaptureButtonClick(View view) {
        imageCapture.takePicture(ContextCompat.getMainExecutor(this), new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                originalBitmap = imageProxyToBitmap(image);
                displayBinarizedImage(originalBitmap);
                image.close();
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e("CameraX", "Capture failed: ", exception);
            }
        });
    }

    public void onUploadButtonClick(View view) {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        pickImageLauncher.launch(intent);
    }

    private void displayBinarizedImage(Bitmap bitmap) {
        Mat originalMat = new Mat();
        Utils.bitmapToMat(bitmap, originalMat);
        Mat grayTemp = new Mat();
        Imgproc.cvtColor(originalMat, grayTemp, Imgproc.COLOR_RGBA2GRAY);
        Mat binaryTemp = new Mat();
        Imgproc.threshold(grayTemp, binaryTemp, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        Bitmap binarizedBitmap = Bitmap.createBitmap(binaryTemp.cols(), binaryTemp.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(binaryTemp, binarizedBitmap);

        showBinarizedImage(binarizedBitmap);

        originalMat.release();
        grayTemp.release();
        binaryTemp.release();
    }

    public void onProcessButtonClick(View view) {
        if (originalBitmap != null) {
            processMaze(originalBitmap);
        }
    }

    private void processMaze(Bitmap bitmap) {
        Mat originalMat = new Mat();
        Utils.bitmapToMat(bitmap, originalMat);

        Mat grayTemp = new Mat();
        Imgproc.cvtColor(originalMat, grayTemp, Imgproc.COLOR_RGBA2GRAY);

        Mat binaryTemp = new Mat();
        Imgproc.threshold(grayTemp, binaryTemp, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

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

        int gridSize = 20;
        int[][] grid = convertToLogicGrid(processedMat, gridSize);

        displayGrid(grid);

        Mat displayMat = new Mat(1000, 1000, CvType.CV_8UC3, new Scalar(255, 255, 255));
        Bitmap gridBitmap = Bitmap.createBitmap(displayMat.cols(), displayMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(displayMat, gridBitmap);

        List<PathFinder.Node> path = PathFinder.findPath(grid, 0, 0, grid.length - 1, grid[0].length - 1);
        drawPathOnBitmap(gridBitmap, path, gridSize);

        originalMat.release();
        grayTemp.release();
        binaryTemp.release();
        processedMat.release();
        displayMat.release();
    }

    private void displayGrid(int[][] grid) {
        StringBuilder gridText = new StringBuilder();
        for (int[] row : grid) {
            for (int cell : row) {
                gridText.append(cell).append(" ");
            }
            gridText.append("\n");
        }
        runOnUiThread(() -> {
            gridTextView.setText(gridText.toString());
            gridScrollView.setVisibility(View.VISIBLE);
        });
    }

    private void drawPathOnBitmap(Bitmap baseBitmap, List<PathFinder.Node> path, int gridSize) {
        Bitmap mutableBitmap = baseBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStrokeWidth(5f);

        if (path != null) {
            for (int i = 0; i < path.size() - 1; i++) {
                float startX = path.get(i).y * gridSize + (gridSize / 2f);
                float startY = path.get(i).x * gridSize + (gridSize / 2f);
                float endX = path.get(i + 1).y * gridSize + (gridSize / 2f);
                float endY = path.get(i + 1).x * gridSize + (gridSize / 2f);
                canvas.drawLine(startX, startY, endX, endY, paint);
            }
        }

        runOnUiThread(() -> processedImageView.setImageBitmap(mutableBitmap));
    }

    private Mat robustClean(Mat warpedMat) {
        Mat gray = new Mat();
        Mat binary = new Mat();

        Imgproc.cvtColor(warpedMat, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

        Core.normalize(gray, gray, 0, 255, Core.NORM_MINMAX);

        Imgproc.threshold(gray, binary, 0, 255, Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernel);

        gray.release();
        kernel.release();

        return binary;
    }

    public int[][] convertToLogicGrid(Mat binaryMat, int gridSize) {
        int rows = binaryMat.rows() / gridSize;
        int cols = binaryMat.cols() / gridSize;
        int[][] logicGrid = new int[rows][cols];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int wallPixelCount = 0;

                for (int i = 0; i < gridSize; i++) {
                    for (int j = 0; j < gridSize; j++) {
                        int checkR = r * gridSize + i;
                        int checkC = c * gridSize + j;

                        if (checkR < binaryMat.rows() && checkC < binaryMat.cols()) {
                            double[] pixel = binaryMat.get(checkR, checkC);
                            if (pixel[0] > 200) {
                                wallPixelCount++;
                            }
                        }
                    }
                }
                logicGrid[r][c] = (wallPixelCount > (gridSize * gridSize * 0.15)) ? 1 : 0;
            }
        }
        return logicGrid;
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
        src.release();
        dst.release();
        return warped;
    }

    private Point[] sortCorners(Point[] corners) {
        List<Point> cornerList = new ArrayList<>(Arrays.asList(corners));
        cornerList.sort(Comparator.comparingDouble(p -> p.y));
        if (cornerList.get(0).x > cornerList.get(1).x) {
            Collections.swap(cornerList, 0, 1);
        }
        if (cornerList.get(2).x < cornerList.get(3).x) {
            Collections.swap(cornerList, 2, 3);
        }

        return cornerList.toArray(new Point[0]);
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
                return result;
            }
            approx.release();
            contour2f.release();
        }

        for (MatOfPoint contour : contours) {
            contour.release();
        }
        hierarchy.release();

        return null;
    }

    private void showBinarizedImage(Bitmap bitmap) {
        previewView.setVisibility(View.GONE);
        captureUploadLayout.setVisibility(View.GONE);
        processedImageView.setVisibility(View.VISIBLE);
        processBackLayout.setVisibility(View.VISIBLE);
        gridScrollView.setVisibility(View.GONE);
        processedImageView.setImageBitmap(bitmap);
    }

    private void showPreview() {
        previewView.setVisibility(View.VISIBLE);
        captureUploadLayout.setVisibility(View.VISIBLE);
        processedImageView.setVisibility(View.GONE);
        processBackLayout.setVisibility(View.GONE);
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
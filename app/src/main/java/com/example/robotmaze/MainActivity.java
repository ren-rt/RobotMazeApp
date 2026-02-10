package com.example.robotmaze;

import android.Manifest;
import android.app.AlertDialog;
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

import androidx.activity.EdgeToEdge;
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
import org.opencv.imgproc.Moments;

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

    private List<Point> detectedEntryPoints = new ArrayList<>();
    private int[][] currentGrid;
    private int currentGridSize;
    private Mat currentProcessedMat;

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
        if (imageCapture == null) return;
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

        Mat warpedOriginal;
        if (corners != null) {
            warpedOriginal = warpMaze(originalMat, corners);
        } else {
            warpedOriginal = originalMat.clone();
        }

        detectedEntryPoints = findGreenMarkers(warpedOriginal);

        Mat processedMat;
        Imgproc.cvtColor(warpedOriginal, warpedOriginal, Imgproc.COLOR_RGBA2BGR);
        processedMat = robustClean(warpedOriginal);

        int targetCells = 200;
        currentGridSize = Math.max(3, Math.min(processedMat.cols(), processedMat.rows()) / targetCells);

        currentGrid = convertToLogicGrid(processedMat, currentGridSize);
        if (currentProcessedMat != null) {
            currentProcessedMat.release();
        }
        currentProcessedMat = processedMat.clone();

        displayGrid(currentGrid);

        if (detectedEntryPoints.isEmpty()) {
            runOnUiThread(() -> Toast.makeText(this, "No green markers detected! Please mark entry/exit points with green dots.", Toast.LENGTH_LONG).show());
            Mat displayMat = new Mat(processedMat.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
            Bitmap gridBitmap = Bitmap.createBitmap(displayMat.cols(), displayMat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(displayMat, gridBitmap);
            drawGridOnBitmap(gridBitmap, currentGrid, currentGridSize);
            processedImageView.setImageBitmap(gridBitmap);
            displayMat.release();
            originalMat.release();
            grayTemp.release();
            binaryTemp.release();
            processedMat.release();
            warpedOriginal.release();
            return;
        }

        if (detectedEntryPoints.size() < 2) {
            runOnUiThread(() -> Toast.makeText(this, "Only " + detectedEntryPoints.size() + " marker(s) detected. Need at least 2!", Toast.LENGTH_LONG).show());
            Mat displayMat = new Mat(processedMat.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
            Bitmap gridBitmap = Bitmap.createBitmap(displayMat.cols(), displayMat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(displayMat, gridBitmap);
            drawGridOnBitmap(gridBitmap, currentGrid, currentGridSize);
            processedImageView.setImageBitmap(gridBitmap);
            displayMat.release();
            originalMat.release();
            grayTemp.release();
            binaryTemp.release();
            processedMat.release();
            warpedOriginal.release();
            return;
        }

        List<Point> gridPoints = new ArrayList<>();
        for (Point p : detectedEntryPoints) {
            int gridX = (int)(p.x / currentGridSize);
            int gridY = (int)(p.y / currentGridSize);
            gridX = Math.max(0, Math.min(gridX, currentGrid.length - 1));
            gridY = Math.max(0, Math.min(gridY, currentGrid[0].length - 1));
            gridPoints.add(new Point(gridX, gridY));
        }

        runOnUiThread(() -> showStartPointSelection(gridPoints));

        originalMat.release();
        grayTemp.release();
        binaryTemp.release();
        processedMat.release();
        warpedOriginal.release();
    }

    private void showStartPointSelection(List<Point> gridPoints) {
        if (gridPoints.size() < 2) {
            Toast.makeText(this, "Need at least 2 entry/exit points!", Toast.LENGTH_LONG).show();
            return;
        }

        String[] options = new String[gridPoints.size()];
        for (int i = 0; i < gridPoints.size(); i++) {
            options[i] = "Point " + (i + 1) + " at grid (" + (int)gridPoints.get(i).x + ", " + (int)gridPoints.get(i).y + ")";
        }

        new AlertDialog.Builder(this)
            .setTitle("Select Starting Point")
            .setItems(options, (dialog, which) -> findBestPath(gridPoints, which))
            .setNegativeButton("Cancel", null)
            .show();
    }

    private void findBestPath(List<Point> gridPoints, int startIndex) {
        Point startPoint = gridPoints.get(startIndex);

        List<PathResult> paths = new ArrayList<>();

        for (int i = 0; i < gridPoints.size(); i++) {
            if (i == startIndex) continue;

            Point endPoint = gridPoints.get(i);
            List<PathFinder.Node> path = PathFinder.findPath(
                currentGrid,
                (int)startPoint.x,
                (int)startPoint.y,
                (int)endPoint.x,
                (int)endPoint.y
            );

            if (path != null && !path.isEmpty()) {
                paths.add(new PathResult(path, startPoint, endPoint, i));
            }
        }

        if (paths.isEmpty()) {
            runOnUiThread(() -> Toast.makeText(this, "No path found from selected starting point!", Toast.LENGTH_LONG).show());
            return;
        }

        paths.sort(Comparator.comparingInt(p -> p.path.size()));
        PathResult shortestPath = paths.get(0);

        final PathResult finalPath = shortestPath;
        runOnUiThread(() -> {
            Toast.makeText(this, "Shortest path: Point " + (startIndex + 1) + " to Point " + (finalPath.endIndex + 1) + " (" + finalPath.path.size() + " steps)", Toast.LENGTH_LONG).show();
            displayPath(finalPath);
        });
    }

    private void displayPath(PathResult pathResult) {
        Mat displayMat = new Mat(currentProcessedMat.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        Bitmap gridBitmap = Bitmap.createBitmap(displayMat.cols(), displayMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(displayMat, gridBitmap);

        drawGridOnBitmap(gridBitmap, currentGrid, currentGridSize);
        drawPathOnBitmap(gridBitmap, pathResult.path, currentGridSize, pathResult.start, pathResult.end);

        displayMat.release();
    }

    private static class PathResult {
        List<PathFinder.Node> path;
        Point start;
        Point end;
        int endIndex;

        PathResult(List<PathFinder.Node> path, Point start, Point end, int endIndex) {
            this.path = path;
            this.start = start;
            this.end = end;
            this.endIndex = endIndex;
        }
    }

    private List<Point> findGreenMarkers(Mat image) {
        List<Point> markers = new ArrayList<>();

        Mat hsvImage = new Mat();
        Imgproc.cvtColor(image, hsvImage, Imgproc.COLOR_RGBA2BGR);
        Imgproc.cvtColor(hsvImage, hsvImage, Imgproc.COLOR_BGR2HSV);

        Scalar lowerGreen = new Scalar(40, 80, 80);
        Scalar upperGreen = new Scalar(80, 255, 255);

        Mat greenMask = new Mat();
        Core.inRange(hsvImage, lowerGreen, upperGreen, greenMask);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.morphologyEx(greenMask, greenMask, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(greenMask, greenMask, Imgproc.MORPH_CLOSE, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(greenMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > 50 && area < 20000) {
                Moments moments = Imgproc.moments(contour);
                if (moments.get_m00() != 0) {
                    int cx = (int)(moments.get_m10() / moments.get_m00());
                    int cy = (int)(moments.get_m01() / moments.get_m00());
                    markers.add(new Point(cy, cx));
                }
            }
            contour.release();
        }

        hsvImage.release();
        greenMask.release();
        kernel.release();
        hierarchy.release();

        return markers;
    }

    private void drawGridOnBitmap(Bitmap baseBitmap, int[][] grid, int gridSize) {
        Canvas canvas = new Canvas(baseBitmap);
        Paint wallPaint = new Paint();
        wallPaint.setColor(Color.BLACK);
        wallPaint.setStyle(Paint.Style.FILL);

        Paint pathPaint = new Paint();
        pathPaint.setColor(Color.WHITE);
        pathPaint.setStyle(Paint.Style.FILL);

        for (int r = 0; r < grid.length; r++) {
            for (int c = 0; c < grid[0].length; c++) {
                Paint paint = (grid[r][c] == 1) ? wallPaint : pathPaint;
                canvas.drawRect(
                    c * gridSize,
                    r * gridSize,
                    (c + 1) * gridSize,
                    (r + 1) * gridSize,
                    paint
                );
            }
        }

        Paint markerPaint = new Paint();
        markerPaint.setColor(Color.GREEN);
        markerPaint.setStyle(Paint.Style.FILL);
        markerPaint.setAlpha(180);

        for (Point marker : detectedEntryPoints) {
            int gridX = (int)(marker.x / gridSize);
            int gridY = (int)(marker.y / gridSize);
            canvas.drawCircle(
                gridY * gridSize + (gridSize / 2f),
                gridX * gridSize + (gridSize / 2f),
                gridSize * 2f,
                markerPaint
            );
        }

        runOnUiThread(() -> processedImageView.setImageBitmap(baseBitmap));
    }

    private void displayGrid(int[][] grid) {
        StringBuilder gridText = new StringBuilder();
        gridText.append("Grid size: ").append(grid.length).append(" x ").append(grid[0].length).append("\n\n");
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

    private void drawPathOnBitmap(Bitmap baseBitmap, List<PathFinder.Node> path, int gridSize, Point entry, Point exit) {
        Canvas canvas = new Canvas(baseBitmap);
        Paint pathPaint = new Paint();
        pathPaint.setColor(Color.RED);
        pathPaint.setStrokeWidth(5f);
        pathPaint.setStyle(Paint.Style.STROKE);

        Paint pointPaint = new Paint();
        pointPaint.setColor(Color.GREEN);
        pointPaint.setStyle(Paint.Style.FILL);

        Paint exitPaint = new Paint();
        exitPaint.setColor(Color.BLUE);
        exitPaint.setStyle(Paint.Style.FILL);

        if (path != null && path.size() > 1) {
            for (int i = 0; i < path.size() - 1; i++) {
                float startX = path.get(i).y * gridSize + (gridSize / 2f);
                float startY = path.get(i).x * gridSize + (gridSize / 2f);
                float endX = path.get(i + 1).y * gridSize + (gridSize / 2f);
                float endY = path.get(i + 1).x * gridSize + (gridSize / 2f);
                canvas.drawLine(startX, startY, endX, endY, pathPaint);
            }

            float entryX = (float)entry.y * gridSize + (gridSize / 2f);
            float entryY = (float)entry.x * gridSize + (gridSize / 2f);
            canvas.drawCircle(entryX, entryY, gridSize / 3f, pointPaint);

            float exitX = (float)exit.y * gridSize + (gridSize / 2f);
            float exitY = (float)exit.x * gridSize + (gridSize / 2f);
            canvas.drawCircle(exitX, exitY, gridSize / 3f, exitPaint);
        }

        runOnUiThread(() -> processedImageView.setImageBitmap(baseBitmap));
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
                int totalPixels = 0;

                for (int i = 0; i < gridSize; i++) {
                    for (int j = 0; j < gridSize; j++) {
                        int checkR = r * gridSize + i;
                        int checkC = c * gridSize + j;

                        if (checkR < binaryMat.rows() && checkC < binaryMat.cols()) {
                            double[] pixel = binaryMat.get(checkR, checkC);
                            if (pixel[0] > 200) {  // White pixel = wall in inverted binary
                                wallPixelCount++;
                            }
                            totalPixels++;
                        }
                    }
                }
                logicGrid[r][c] = (wallPixelCount > (totalPixels * 0.25)) ? 1 : 0;
            }
        }

        Log.d(TAG, "Logic grid created: " + rows + " x " + cols);
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
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
import android.os.Build;
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
    private static final int BLUETOOTH_PERMISSION_REQUEST_CODE = 101;

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

    private final String DEVICE_ADDRESS = "4C:03:B3:F7:24:ED"; // Replace with your device's address
    private Runnable onBluetoothPermissionGranted;

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

    private List<Point> detectedEntryPoints = new ArrayList<>();
    private int[][] currentGrid;
    private int currentGridSize;
    private Mat currentProcessedMat;

    private void processMaze(Bitmap bitmap) {
        Mat originalMat = new Mat();
        Utils.bitmapToMat(bitmap, originalMat);

        Mat originalBGR = new Mat();
        Imgproc.cvtColor(originalMat, originalBGR, Imgproc.COLOR_RGBA2BGR);
        detectedEntryPoints = findGreenMarkers(originalBGR);

        Mat grayTemp = new Mat();
        Imgproc.cvtColor(originalMat, grayTemp, Imgproc.COLOR_RGBA2GRAY);

        Mat binaryTemp = new Mat();
        Imgproc.threshold(grayTemp, binaryTemp, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        Point[] corners = findMazeCorners(binaryTemp);

        Mat warpedOriginal;
        if (corners != null) {
            warpedOriginal = warpMaze(originalMat, corners);
            List<Point> warpedMarkers = warpPoints(detectedEntryPoints, corners, 1000, 1000);
            detectedEntryPoints = warpedMarkers;
        } else {
            warpedOriginal = originalMat.clone();
        }

        Mat processedMat;
        Imgproc.cvtColor(warpedOriginal, warpedOriginal, Imgproc.COLOR_RGBA2BGR);
        processedMat = robustClean(warpedOriginal);
        originalBGR.release();

        int targetCells = 150;
        currentGridSize = Math.max(4, Math.min(processedMat.cols(), processedMat.rows()) / targetCells);

        currentGrid = convertToLogicGrid(processedMat, currentGridSize);
        if (currentProcessedMat != null) {
            currentProcessedMat.release();
        }
        currentProcessedMat = processedMat.clone();

        displayGrid(currentGrid);

        if (detectedEntryPoints.size() < 2) {
            runOnUiThread(() -> Toast.makeText(this, "Need at least 2 green markers for entry/exit.", Toast.LENGTH_LONG).show());
            return;
        }

        List<Point> gridPoints = new ArrayList<>();
        for (Point p : detectedEntryPoints) {
            int gridX = (int) (p.x / currentGridSize);
            int gridY = (int) (p.y / currentGridSize);
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
        String[] options = new String[gridPoints.size()];
        for (int i = 0; i < gridPoints.size(); i++) {
            options[i] = "Point " + (i + 1) + " at grid (" + (int) gridPoints.get(i).x + ", " + (int) gridPoints.get(i).y + ")";
        }

        new android.app.AlertDialog.Builder(this)
                .setTitle("Select Starting Point")
                .setItems(options, (dialog, which) -> initiatePathfindingAndBluetooth(gridPoints, which))
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void initiatePathfindingAndBluetooth(List<Point> gridPoints, int startIndex) {
        onBluetoothPermissionGranted = () -> findBestPath(gridPoints, startIndex);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED ||
                    checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.BLUETOOTH_SCAN}, BLUETOOTH_PERMISSION_REQUEST_CODE);
            } else {
                onBluetoothPermissionGranted.run();
                onBluetoothPermissionGranted = null;
            }
        } else {
            onBluetoothPermissionGranted.run();
            onBluetoothPermissionGranted = null;
        }
    }

    private void findBestPath(List<Point> gridPoints, int startIndex) {
        Point startPoint = gridPoints.get(startIndex);
        List<PathResult> paths = new ArrayList<>();

        for (int i = 0; i < gridPoints.size(); i++) {
            if (i == startIndex) continue;

            Point endPoint = gridPoints.get(i);
            List<PathFinder.Node> path = PathFinder.findPath(currentGrid, (int) startPoint.x, (int) startPoint.y, (int) endPoint.x, (int) endPoint.y);

            if (path != null && !path.isEmpty() && isPathValid(path, currentGrid)) {
                paths.add(new PathResult(path, startPoint, endPoint, i));
            }
        }

        if (paths.isEmpty()) {
            runOnUiThread(() -> Toast.makeText(this, "No valid path found from selected starting point!", Toast.LENGTH_LONG).show());
            return;
        }

        PathResult shortestPath = Collections.min(paths, Comparator.comparingInt(p -> p.path.size()));

        runOnUiThread(() -> {
            displayPath(shortestPath);
            launchBluetoothActivity(shortestPath);
        });
    }

    private void launchBluetoothActivity(PathResult pathResult) {
        StringBuilder pathString = new StringBuilder();
        for (PathFinder.Node node : pathResult.path) {
            pathString.append(node.x).append(",").append(node.y).append(";");
        }

        Intent intent = new Intent(MainActivity.this, BluetoothCommandActivity.class);
        intent.putExtra("device_address", DEVICE_ADDRESS);
        intent.putExtra("path", pathString.toString());
        startActivity(intent);
    }

    private boolean isPathValid(List<PathFinder.Node> path, int[][] grid) {
        for (PathFinder.Node node : path) {
            if (grid[node.x][node.y] == 1) return false;
        }
        return true;
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
        Point start, end;
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
        Imgproc.cvtColor(image, hsvImage, Imgproc.COLOR_BGR2HSV);
        Scalar lowerGreen = new Scalar(45, 100, 100);
        Scalar upperGreen = new Scalar(75, 255, 255);
        Mat greenMask = new Mat();
        Core.inRange(hsvImage, lowerGreen, upperGreen, greenMask);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
        Imgproc.morphologyEx(greenMask, greenMask, Imgproc.MORPH_OPEN, kernel, new Point(-1, -1), 2);
        Imgproc.dilate(greenMask, greenMask, kernel, new Point(-1, -1), 1);
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(greenMask, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            if (Imgproc.contourArea(contour) > 100) {
                Moments moments = Imgproc.moments(contour);
                if (moments.get_m00() != 0) {
                    int cx = (int) (moments.get_m10() / moments.get_m00());
                    int cy = (int) (moments.get_m01() / moments.get_m00());
                    markers.add(new Point(cy, cx));
                }
            }
            contour.release();
        }
        hsvImage.release();
        greenMask.release();
        kernel.release();
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
                canvas.drawRect(c * gridSize, r * gridSize, (c + 1) * gridSize, (r + 1) * gridSize, paint);
            }
        }

        Paint markerPaint = new Paint();
        markerPaint.setColor(Color.GREEN);
        markerPaint.setStyle(Paint.Style.FILL);
        markerPaint.setAlpha(180);
        for (Point marker : detectedEntryPoints) {
            int gridX = (int) (marker.x / gridSize);
            int gridY = (int) (marker.y / gridSize);
            canvas.drawCircle(gridY * gridSize + (gridSize / 2f), gridX * gridSize + (gridSize / 2f), gridSize * 2f, markerPaint);
        }
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
            canvas.drawCircle((float) entry.y * gridSize + (gridSize / 2f), (float) entry.x * gridSize + (gridSize / 2f), gridSize / 3f, pointPaint);
            canvas.drawCircle((float) exit.y * gridSize + (gridSize / 2f), (float) exit.x * gridSize + (gridSize / 2f), gridSize / 3f, exitPaint);
        }
        runOnUiThread(() -> processedImageView.setImageBitmap(baseBitmap));
    }

    private Mat robustClean(Mat warpedMat) {
        Mat gray = new Mat();
        Imgproc.cvtColor(warpedMat, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);
        Core.normalize(gray, gray, 0, 255, Core.NORM_MINMAX);
        Mat binary = new Mat();
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
                Mat cell = new Mat(binaryMat, new org.opencv.core.Rect(c * gridSize, r * gridSize, gridSize, gridSize));
                int wallPixels = Core.countNonZero(cell);
                logicGrid[r][c] = (wallPixels > (gridSize * gridSize * 0.4)) ? 1 : 0;
                cell.release();
            }
        }
        return logicGrid;
    }

    private Mat warpMaze(Mat input, Point[] corners) {
        Point[] sortedCorners = sortCorners(corners);
        MatOfPoint2f src = new MatOfPoint2f(sortedCorners);
        MatOfPoint2f dst = new MatOfPoint2f(new Point(0, 0), new Point(1000, 0), new Point(1000, 1000), new Point(0, 1000));
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
        if (cornerList.get(0).x > cornerList.get(1).x) Collections.swap(cornerList, 0, 1);
        if (cornerList.get(2).x < cornerList.get(3).x) Collections.swap(cornerList, 2, 3);
        return cornerList.toArray(new Point[0]);
    }

    private List<Point> warpPoints(List<Point> points, Point[] corners, int dstWidth, int dstHeight) {
        if (points.isEmpty()) return points;
        MatOfPoint2f src = new MatOfPoint2f(sortCorners(corners));
        MatOfPoint2f dst = new MatOfPoint2f(new Point(0, 0), new Point(dstWidth, 0), new Point(dstWidth, dstHeight), new Point(0, dstHeight));
        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(src, dst);
        List<Point> warpedPoints = new ArrayList<>();
        for (Point p : points) {
            MatOfPoint2f pointMat = new MatOfPoint2f(p);
            MatOfPoint2f warpedPointMat = new MatOfPoint2f();
            Core.perspectiveTransform(pointMat, warpedPointMat, perspectiveTransform);
            warpedPoints.add(warpedPointMat.toArray()[0]);
            pointMat.release();
            warpedPointMat.release();
        }
        perspectiveTransform.release();
        src.release();
        dst.release();
        return warpedPoints;
    }

    private Point[] findMazeCorners(Mat binary) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(binary, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        MatOfPoint largestContour = null;
        double maxArea = 0;
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
                imageCapture = new ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build();
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture);
            } catch (Exception e) {
                Log.e("CameraX", "Binding failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
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
        } else if (requestCode == BLUETOOTH_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                if (onBluetoothPermissionGranted != null) {
                    onBluetoothPermissionGranted.run();
                    onBluetoothPermissionGranted = null;
                }
            } else {
                Toast.makeText(this, "Bluetooth permissions are required to connect to the robot.", Toast.LENGTH_SHORT).show();
            }
        }
    }
}

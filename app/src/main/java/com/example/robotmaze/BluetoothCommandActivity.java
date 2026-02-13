package com.example.robotmaze;

import android.Manifest;
import android.annotation.SuppressLint;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothSocket;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Set;
import java.util.UUID;

public class BluetoothCommandActivity extends AppCompatActivity {

    private static final String TAG = "BluetoothCommand";
    // Standard UUID for HC-05 SPP (Serial Port Profile)
    private static final UUID HC05_UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB");
    private static final int BLUETOOTH_PERMISSION_REQUEST = 102;

    private BluetoothAdapter bluetoothAdapter;
    private BluetoothSocket bluetoothSocket;
    private OutputStream outputStream;
    private String deviceAddress;
    private String pathData;

    private TextView statusText;
    private TextView pathText;
    private TextView receivedText;
    private Button scanButton, connectButton, sendPathButton, disconnectButton, sendCommandButton;
    private EditText commandInput;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_bluetooth_command);

        deviceAddress = getIntent().getStringExtra("device_address");
        pathData = getIntent().getStringExtra("path");

        initializeUI();
        pathText.setText(pathData);

        final BluetoothManager bluetoothManager = getSystemService(BluetoothManager.class);
        bluetoothAdapter = bluetoothManager.getAdapter();

        updateStatus("Ready");
    }

    private void initializeUI() {
        statusText = findViewById(R.id.status_text);
        pathText = findViewById(R.id.path_text);
        scanButton = findViewById(R.id.scan_button);
        connectButton = findViewById(R.id.connect_button);
        sendPathButton = findViewById(R.id.send_path_button);
        disconnectButton = findViewById(R.id.disconnect_button);
        receivedText = findViewById(R.id.received_text);
        commandInput = findViewById(R.id.command_input);
        sendCommandButton = findViewById(R.id.send_command_button);

        scanButton.setOnClickListener(v -> listPairedDevices());
        connectButton.setOnClickListener(v -> connectToDevice());
        sendPathButton.setOnClickListener(v -> sendPath());
        disconnectButton.setOnClickListener(v -> disconnect());
        sendCommandButton.setOnClickListener(v -> sendCustomCommand());
    }

    @SuppressLint("MissingPermission")
    private void listPairedDevices() {
        if (!checkPermissions()) return;
        receivedText.setText("Paired Devices:\n");
        Set<BluetoothDevice> pairedDevices = bluetoothAdapter.getBondedDevices();
        if (pairedDevices.size() > 0) {
            for (BluetoothDevice device : pairedDevices) {
                receivedText.append("\n" + device.getName() + "\n" + device.getAddress());
            }
        } else {
            receivedText.setText("No paired devices found.");
        }
    }

    @SuppressLint("MissingPermission")
    private void connectToDevice() {
        if (!checkPermissions() || bluetoothAdapter == null) return;
        if (!bluetoothAdapter.isEnabled()) {
            updateStatus("Please enable Bluetooth");
            return;
        }

        updateStatus("Connecting to " + deviceAddress + "...");
        connectButton.setEnabled(false);

        new Thread(() -> {
            try {
                BluetoothDevice device = bluetoothAdapter.getRemoteDevice(deviceAddress);
                bluetoothSocket = device.createRfcommSocketToServiceRecord(HC05_UUID);
                bluetoothSocket.connect();
                outputStream = bluetoothSocket.getOutputStream();
                runOnUiThread(() -> {
                    updateStatus("Connected to " + device.getName());
                    sendPathButton.setEnabled(true);
                    disconnectButton.setEnabled(true);
                    sendCommandButton.setEnabled(true);
                    connectButton.setEnabled(false);
                });
            } catch (IOException e) {
                Log.e(TAG, "Connection failed", e);
                runOnUiThread(() -> {
                    updateStatus("Connection failed. Is device paired and in range?");
                    connectButton.setEnabled(true);
                });
                disconnect(); // Clean up on failure
            }
        }).start();
    }

    private void sendPath() {
        sendBluetoothMessage(pathData);
    }

    private void sendCustomCommand() {
        String command = commandInput.getText().toString();
        if (!command.isEmpty()) {
            sendBluetoothMessage(command);
            commandInput.setText("");
        }
    }

    private void sendBluetoothMessage(String message) {
        if (outputStream == null) {
            updateStatus("Not connected");
            return;
        }
        new Thread(() -> {
            try {
                outputStream.write((message + "\n").getBytes()); // Add newline as a delimiter
                outputStream.flush();
                Log.d(TAG, "Sent: " + message);
            } catch (IOException e) {
                Log.e(TAG, "Send failed", e);
                runOnUiThread(() -> updateStatus("Send failed"));
            }
        }).start();
    }

    private void disconnect() {
        try {
            if (outputStream != null) outputStream.close();
            if (bluetoothSocket != null) bluetoothSocket.close();
        } catch (IOException e) {
            Log.e(TAG, "Error during disconnect", e);
        }
        outputStream = null;
        bluetoothSocket = null;
        runOnUiThread(() -> {
            updateStatus("Disconnected");
            sendPathButton.setEnabled(false);
            disconnectButton.setEnabled(false);
            connectButton.setEnabled(true);
            sendCommandButton.setEnabled(false);
        });
    }

    private void updateStatus(String message) {
        runOnUiThread(() -> statusText.setText("Status: " + message));
        Log.d(TAG, message);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        disconnect();
    }

    private boolean checkPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.BLUETOOTH_SCAN}, BLUETOOTH_PERMISSION_REQUEST);
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == BLUETOOTH_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                updateStatus("Permissions granted.");
            } else {
                updateStatus("Permissions denied.");
                Toast.makeText(this, "Bluetooth permissions are required.", Toast.LENGTH_SHORT).show();
            }
        }
    }
}

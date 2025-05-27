// components/AttendanceScanner.js
import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Alert,
  ScrollView,
  Dimensions
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Camera } from 'expo-camera';

const { width, height } = Dimensions.get('window');

export default function AttendanceScanner({ route, navigation }) {
  const { classData } = route.params;
  const [hasPermission, setHasPermission] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [attendanceRecords, setAttendanceRecords] = useState([]);
  const [loading, setLoading] = useState(false);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const captureAndRecognize = async () => {
    if (!cameraRef.current) return;

    setLoading(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.8,
      });

      // Mock face recognition - replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockResult = {
        success: Math.random() > 0.3,
        student_id: 'ST001',
        student_name: 'John Doe',
        confidence: 0.95,
      };

      if (mockResult.success) {
        const newRecord = {
          student_id: mockResult.student_id,
          student_name: mockResult.student_name,
          timestamp: new Date().toLocaleTimeString(),
          confidence: mockResult.confidence,
        };
        
        setAttendanceRecords(prev => [newRecord, ...prev]);
        Alert.alert('Success', `${mockResult.student_name} marked present!`);
      } else {
        Alert.alert('Not Found', 'No matching student found. Please try again.');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to capture image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <Text>Requesting camera permission...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Text>No access to camera</Text>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.buttonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Take Attendance</Text>
        <Text style={styles.subtitle}>{attendanceRecords.length} students marked</Text>
      </View>

      <ScrollView style={styles.content}>
        {/* Camera Section */}
        <View style={styles.cameraContainer}>
          {isScanning ? (
            <View style={styles.cameraWrapper}>
              <Camera
                ref={cameraRef}
                style={styles.camera}
                type={Camera.Constants.Type.front}
              />
              <View style={styles.overlay}>
                <View style={styles.scanFrame} />
              </View>
            </View>
          ) : (
            <View style={styles.cameraPlaceholder}>
              <Text style={styles.placeholderText}>📷</Text>
              <Text style={styles.placeholderSubtext}>
                Tap "Start Camera" to begin scanning
              </Text>
            </View>
          )}
        </View>

        {/* Control Buttons */}
        <View style={styles.controls}>
          {!isScanning ? (
            <TouchableOpacity
              style={styles.startButton}
              onPress={() => setIsScanning(true)}
            >
              <Text style={styles.startButtonText}>Start Camera</Text>
            </TouchableOpacity>
          ) : (
            <View style={styles.scanControls}>
              <TouchableOpacity
                style={[styles.captureButton, loading && styles.disabledButton]}
                onPress={captureAndRecognize}
                disabled={loading}
              >
                <Text style={styles.captureButtonText}>
                  {loading ? 'Scanning...' : 'Capture & Recognize'}
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.stopButton}
                onPress={() => setIsScanning(false)}
              >
                <Text style={styles.stopButtonText}>Stop Camera</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>

        {/* Attendance Records */}
        <View style={styles.recordsSection}>
          <Text style={styles.recordsTitle}>Today's Attendance</Text>
          {attendanceRecords.length === 0 ? (
            <Text style={styles.noRecords}>No students marked present yet</Text>
          ) : (
            attendanceRecords.map((record, index) => (
              <View key={index} style={styles.recordCard}>
                <View style={styles.recordInfo}>
                  <Text style={styles.recordName}>{record.student_name}</Text>
                  <Text style={styles.recordId}>ID: {record.student_id}</Text>
                </View>
                <View style={styles.recordTime}>
                  <Text style={styles.timeText}>{record.timestamp}</Text>
                  <Text style={styles.confidenceText}>
                    {Math.round(record.confidence * 100)}% confidence
                  </Text>
                </View>
              </View>
            ))
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    backgroundColor: '#fff',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
    alignItems: 'center',
  },
  backButton: {
    alignSelf: 'flex-start',
    marginBottom: 10,
  },
  backButtonText: {
    color: '#3498db',
    fontSize: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  subtitle: {
    fontSize: 14,
    color: '#7f8c8d',
    marginTop: 5,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  cameraContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
    height: 300,
  },
  cameraWrapper: {
    flex: 1,
    position: 'relative',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scanFrame: {
    width: 200,
    height: 200,
    borderWidth: 2,
    borderColor: '#3498db',
    borderRadius: 12,
    backgroundColor: 'transparent',
  },
  cameraPlaceholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f8f9fa',
  },
  placeholderText: {
    fontSize: 48,
    marginBottom: 10,
  },
  placeholderSubtext: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
  },
  controls: {
    marginBottom: 20,
  },
  startButton: {
    backgroundColor: '#3498db',
    borderRadius: 12,
    padding: 15,
    alignItems: 'center',
  },
  startButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  scanControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  captureButton: {
    backgroundColor: '#27ae60',
    borderRadius: 12,
    padding: 15,
    flex: 1,
    marginRight: 10,
    alignItems: 'center',
  },
  captureButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  stopButton: {
    backgroundColor: '#e74c3c',
    borderRadius: 12,
    padding: 15,
    flex: 1,
    marginLeft: 10,
    alignItems: 'center',
  },
  stopButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  disabledButton: {
    opacity: 0.6,
  },
  recordsSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
  },
  recordsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 15,
  },
  noRecords: {
    textAlign: 'center',
    color: '#7f8c8d',
    fontSize: 14,
    paddingVertical: 20,
  },
  recordCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#e8f5e8',
    borderRadius: 8,
    padding: 12,
    marginBottom: 10,
  },
  recordInfo: {
    flex: 1,
  },
  recordName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
  },
  recordId: {
    fontSize: 12,
    color: '#7f8c8d',
    marginTop: 2,
  },
  recordTime: {
    alignItems: 'flex-end',
  },
  timeText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#27ae60',
  },
  confidenceText: {
    fontSize: 10,
    color: '#7f8c8d',
    marginTop: 2,
  },
});
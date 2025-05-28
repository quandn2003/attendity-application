// components/AttendanceScanner.js
import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  Platform
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import ApiService from '../services/apiService';

export default function AttendanceScanner({ route, navigation }) {
  console.log('=== AttendanceScanner START ===');
  console.log('Component initialized');
  console.log('Route params:', route?.params);
  console.log('Navigation object:', navigation);
  
  const { classData } = route.params;
  console.log('Class data extracted:', classData);
  
  const [isScanning, setIsScanning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [attendanceRecords, setAttendanceRecords] = useState([]);
  const [facing, setFacing] = useState('front');
  
  // Use the new useCameraPermissions hook
  const [permission, requestPermission] = useCameraPermissions();
  
  console.log('Initial state set');
  console.log('isScanning:', isScanning);
  console.log('loading:', loading);
  console.log('permission:', permission);
  
  const camera = useRef(null);
  console.log('Camera ref created:', camera);

  const [isProcessing, setIsProcessing] = useState(false);
  const [lastScanTime, setLastScanTime] = useState(0);

  const convertImageToBase64 = async (uri) => {
    try {
      console.log('Converting image to base64 from URI:', uri);
      const response = await fetch(uri);
      console.log('Fetch response status:', response.status);
      console.log('Fetch response ok:', response.ok);
      
      const blob = await response.blob();
      console.log('Blob created, size:', blob.size, 'type:', blob.type);
      
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          console.log('FileReader finished reading');
          const base64String = reader.result.split(',')[1];
          console.log('Base64 string extracted, length:', base64String.length);
          resolve(base64String);
        };
        reader.onerror = (error) => {
          console.error('FileReader error:', error);
          reject(error);
        };
        reader.readAsDataURL(blob);
      });
    } catch (error) {
      console.error('Error converting image to base64:', error);
      throw error;
    }
  };

  const captureAndRecognize = async () => {
    console.log('=== captureAndRecognize START ===');
    console.log('Camera ref current:', camera.current);
    console.log('Has permission check:', permission?.granted);
    
    if (!camera.current || !permission?.granted) {
      console.log('FAIL: Camera not available or no permission');
      Alert.alert('Error', 'Camera not available');
      return;
    }

    // Prevent rapid successive scans
    const now = Date.now();
    if (now - lastScanTime < 3000) {
      Alert.alert('Please wait', 'Please wait a moment before scanning again');
      return;
    }
    setLastScanTime(now);

    console.log('Starting photo capture process...');
    setLoading(true);
    setIsProcessing(true);
    
    try {
      console.log('Calling takePictureAsync...');
      const photo = await camera.current.takePictureAsync({
        quality: 0.9, // High quality for better anti-spoofing
        base64: false,
        skipProcessing: false,
      });
      console.log('Photo captured successfully:', photo);

      // Convert image to base64
      console.log('Converting image to base64...');
      const base64Image = await convertImageToBase64(photo.uri);
      console.log('Image converted to base64, length:', base64Image.length);

      // Step 1: Get face embedding from AI API
      console.log('Getting face embedding...');
      console.log('Sending request to:', `${ApiService.API_CONFIG?.AI_API_URL || 'undefined'}/inference`);
      
      const inferenceResult = await ApiService.performInference(base64Image);
      console.log('=== FULL INFERENCE RESULT ===');
      console.log(JSON.stringify(inferenceResult, null, 2));
      console.log('=== END INFERENCE RESULT ===');
      
      console.log('Inference result keys:', Object.keys(inferenceResult));
      console.log('Has embedding:', !!inferenceResult.embedding);
      console.log('Embedding length:', inferenceResult.embedding?.length);
      console.log('Is real face (is_real):', inferenceResult.is_real);
      console.log('Face detected:', inferenceResult.face_detected);
      console.log('Quality score:', inferenceResult.quality_score);
      console.log('Anti-spoofing confidence:', inferenceResult.confidence);

      // Check if face was detected and is valid
      if (!inferenceResult.embedding || inferenceResult.is_real === false) {
        console.log('Face detection failed:');
        console.log('- Has embedding:', !!inferenceResult.embedding);
        console.log('- Is real face (is_real):', inferenceResult.is_real);
        console.log('- Face detected:', inferenceResult.face_detected);
        console.log('- Anti-spoofing confidence:', inferenceResult.confidence);
        
        Alert.alert(
          'Face Detection Failed', 
          inferenceResult.is_real === false 
            ? 'Please use a real face, not a photo or screen'
            : 'No face detected. Please ensure your face is clearly visible and try again.'
        );
        return;
      }

      // Log the embedding vector
      console.log('=== FACE EMBEDDING VECTOR ===');
      console.log('Embedding first 10 values:', inferenceResult.embedding.slice(0, 10));
      console.log('Embedding length:', inferenceResult.embedding.length);
      console.log('=== END EMBEDDING VECTOR ===');

      // Step 2: Search for matching student using the voting system
      console.log('Searching for student match with voting...');
      const searchResult = await ApiService.searchStudentWithVoting(classData.class_code, inferenceResult.embedding);
      console.log('=== SEARCH RESULT ===');
      console.log(JSON.stringify(searchResult, null, 2));
      console.log('=== END SEARCH RESULT ===');

      if (searchResult.status === 'match_found') {
        // Clear match found - mark attendance
        console.log('Clear match found:', searchResult.student_id);
        
        try {
          const attendanceResult = await ApiService.markAttendance(classData.class_code, searchResult.student_id);
          console.log('Attendance marked successfully:', attendanceResult);
          
          const attendanceRecord = {
            id: Date.now().toString(),
            student_id: searchResult.student_id,
            student_name: searchResult.student_name || searchResult.student_id,
            confidence: searchResult.confidence || 0,
            timestamp: new Date().toLocaleTimeString(),
            photo_path: photo.uri,
            status: 'present'
          };

          setAttendanceRecords(prev => [attendanceRecord, ...prev]);
          Alert.alert('Success!', `${attendanceRecord.student_name} marked present`);
          
        } catch (attendanceError) {
          console.error('Error marking attendance:', attendanceError);
          Alert.alert('Error', 'Failed to mark attendance. Please try again.');
        }
        
      } else if (searchResult.status === 'ambiguous_match') {
        // Ambiguous match - ask for user confirmation
        console.log('Ambiguous match found');
        
        Alert.alert(
          'Confirm Identity',
          `Is this ${searchResult.student_id}?\nConfidence: ${(searchResult.confidence * 100).toFixed(1)}%`,
          [
            { text: 'No', style: 'cancel' },
            { 
              text: 'Yes', 
              onPress: async () => {
                try {
                  await ApiService.markAttendance(classData.class_code, searchResult.student_id);
                  const attendanceRecord = {
                    id: Date.now().toString(),
                    student_id: searchResult.student_id,
                    student_name: searchResult.student_name || searchResult.student_id,
                    confidence: searchResult.confidence || 0,
                    timestamp: new Date().toLocaleTimeString(),
                    photo_path: photo.uri,
                    status: 'present'
                  };

                  setAttendanceRecords(prev => [attendanceRecord, ...prev]);
                  Alert.alert('Success!', `${attendanceRecord.student_name} marked present (confirmed)`);
                } catch (error) {
                  Alert.alert('Error', 'Failed to mark attendance. Please try again.');
                }
              }
            }
          ]
        );
        
      } else {
        // No match found or below threshold
        console.log('No clear match found');
        console.log('Search status:', searchResult.status);
        console.log('Reason:', searchResult.reason);
        
        let message = 'No matching student found in this class.';
        
        if (searchResult.voting_details?.best_similarity !== undefined) {
          const similarity = searchResult.voting_details.best_similarity;
          const threshold = searchResult.voting_details.threshold || 0.6;
          
          if (similarity < 0) {
            message = 'Face recognition failed - the captured image does not match any enrolled students. Please ensure:\n\n• Your face is clearly visible\n• Good lighting conditions\n• You are enrolled in this class\n• Try capturing again';
          } else if (similarity < threshold) {
            message = `Low similarity match (${(similarity * 100).toFixed(1)}% vs required ${(threshold * 100).toFixed(1)}%). Please:\n\n• Ensure good lighting\n• Face the camera directly\n• Remove any obstructions\n• Try again`;
          }
        }
        
        Alert.alert(
          'Student Not Found',
          message,
          [
            { text: 'Try Again', style: 'default' },
            { 
              text: 'Manual Entry', 
              onPress: () => {
                Alert.alert('Manual Entry', 'Manual attendance entry feature coming soon');
              }
            }
          ]
        );
      }
      
    } catch (error) {
      console.error('ERROR in photo capture/recognition:', error);
      console.error('Error message:', error.message);
      
      let errorMessage = 'Failed to process attendance';
      let errorTitle = 'Error';
      
      // Handle specific error types
      if (error.message.includes('Spoofing detected') || error.message.includes('combined_score_too_low')) {
        errorTitle = 'Anti-Spoofing Protection';
        errorMessage = 'The system detected this might not be a live face. Please try again with:\n\n• Better lighting conditions\n• Hold the camera steady\n• Face the camera directly\n• Ensure you are using a real face (not a photo)\n• Move closer to the camera\n\nIf this continues, try the manual entry option.';
        
        Alert.alert(
          errorTitle,
          errorMessage,
          [
            { text: 'Try Again', style: 'default' },
            { 
              text: 'Manual Entry', 
              onPress: () => {
                Alert.alert('Manual Entry', 'Manual attendance entry feature coming soon');
              }
            }
          ]
        );
        return;
      }
      
      // Handle other error types
      if (error.message.includes('network') || error.message.includes('fetch')) {
        errorMessage = 'Network error. Please check your connection and try again.';
      } else if (error.message.includes('timeout')) {
        errorMessage = 'Request timed out. Please try again.';
      } else if (error.message.includes('404')) {
        errorMessage = 'Service not available. Please contact administrator.';
      }
      
      Alert.alert(errorTitle, errorMessage);
    } finally {
      console.log('Setting loading to false');
      setLoading(false);
      setIsProcessing(false);
    }
    console.log('=== captureAndRecognize END ===');
  };

  const renderCamera = () => {
    console.log('=== renderCamera START ===');
    console.log('permission value:', permission);
    
    if (!permission) {
      console.log('Permission is null - showing loading state');
      return (
        <View style={styles.cameraPlaceholder}>
          <ActivityIndicator size="large" color="#3498db" />
          <Text style={styles.placeholderSubtext}>Loading camera permissions...</Text>
        </View>
      );
    }

    if (!permission.granted) {
      console.log('Permission denied - showing permission request UI');
      return (
        <View style={styles.cameraPlaceholder}>
          <Text style={styles.placeholderText}>📷</Text>
          <Text style={styles.placeholderSubtext}>
            Camera permission required
          </Text>
          <TouchableOpacity 
            style={styles.permissionButton}
            onPress={() => {
              console.log('Permission button pressed - requesting permission');
              requestPermission();
            }}
          >
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      );
    }

    console.log('Rendering actual camera component');
    return (
      <CameraView
        ref={camera}
        style={styles.camera}
        facing={facing}
      >
        <View style={styles.overlay}>
          <View style={styles.scanFrame} />
          <Text style={styles.scanText}>Position face within the frame</Text>
        </View>
        
        {/* Processing overlay */}
        {isProcessing && (
          <View style={styles.processingOverlay}>
            <ActivityIndicator size="large" color="#ffffff" />
            <Text style={styles.processingText}>
              Processing face recognition...
            </Text>
          </View>
        )}
      </CameraView>
    );
  };

  console.log('=== Main render START ===');
  console.log('Current state - isScanning:', isScanning, 'loading:', loading);
  console.log('Attendance records count:', attendanceRecords.length);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity 
            style={styles.backButton}
            onPress={() => {
              console.log('Back button pressed - navigating back');
              navigation.goBack();
            }}
          >
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.title}>Take Attendance</Text>
          <Text style={styles.subtitle}>{classData.class_code}</Text>
        </View>

        <View style={styles.content}>
          {/* Camera Section */}
          <View style={styles.cameraContainer}>
            {isScanning ? (
              <View style={styles.cameraWrapper}>
                {renderCamera()}
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
                style={[styles.startButton, !permission?.granted && styles.disabledButton]}
                onPress={() => {
                  console.log('=== START CAMERA BUTTON PRESSED ===');
                  console.log('permission:', permission);
                  if (permission?.granted) {
                    console.log('Permission granted - setting isScanning to true');
                    setIsScanning(true);
                    console.log('isScanning should now be true');
                  } else {
                    console.log('Permission not granted - cannot start camera');
                  }
                }}
                disabled={!permission?.granted}
              >
                <Text style={styles.startButtonText}>
                  {!permission?.granted ? 'Camera Permission Required' : 'Start Camera'}
                </Text>
              </TouchableOpacity>
            ) : (
              <View style={styles.scanControls}>
                <TouchableOpacity
                  style={[styles.captureButton, loading && styles.disabledButton]}
                  onPress={() => {
                    console.log('Capture button pressed - calling captureAndRecognize');
                    captureAndRecognize();
                  }}
                  disabled={loading}
                >
                  <Text style={styles.captureButtonText}>
                    {loading ? 'Scanning...' : 'Capture & Recognize'}
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.stopButton}
                  onPress={() => {
                    console.log('Stop camera button pressed - setting isScanning to false');
                    setIsScanning(false);
                  }}
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
              attendanceRecords.map((record, index) => {
                console.log('Rendering attendance record:', record, 'at index:', index);
                return (
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
                );
              })
            )}
          </View>
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
    backgroundColor: '#000',
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
  scanText: {
    color: '#fff',
    fontSize: 14,
    marginTop: 20,
    textAlign: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 10,
    borderRadius: 8,
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
    marginBottom: 20,
  },
  permissionButton: {
    backgroundColor: '#3498db',
    borderRadius: 8,
    padding: 12,
    paddingHorizontal: 20,
  },
  permissionButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
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
  processingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  processingText: {
    color: 'white',
    fontSize: 16,
    marginTop: 10,
    textAlign: 'center',
  },
});
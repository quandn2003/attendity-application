import React, { useState, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
  Image,
  ScrollView
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Camera, CameraType } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import ApiService from '../services/apiService';
import { Card, Title } from 'react-native-paper';
import { Button } from 'react-native-paper';
import * as FileSystem from 'expo-file-system';

export default function StudentRegistration({ route, navigation }) {
  const { classData } = route.params;
  const [studentId, setStudentId] = useState('');
  const [studentName, setStudentName] = useState('');
  const [studentEmail, setStudentEmail] = useState('');
  const [photos, setPhotos] = useState([null, null, null]);
  const [loading, setLoading] = useState(false);
  const [cameraVisible, setCameraVisible] = useState(false);
  const [currentPhotoIndex, setCurrentPhotoIndex] = useState(0);
  const [imageBase64, setImageBase64] = useState('');
  const [hasPermission, setHasPermission] = useState(null);
  const cameraRef = useRef(null);

  React.useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const convertImageToBase64 = async (uri) => {
    try {
      console.log('🔄 Converting image to base64:', uri);
      const base64 = await FileSystem.readAsStringAsync(uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      console.log('✅ Base64 conversion successful, length:', base64.length);
      return base64;
    } catch (error) {
      console.error('❌ Error converting image to base64:', error);
      throw error;
    }
  };

  const takePhoto = async (index) => {
    try {
      console.log('🎥 Taking photo for index:', index);
      
      if (cameraRef.current) {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          base64: true,
        });
        
        console.log('📸 Photo taken:', photo.uri);
        
        const newPhotos = [...photos];
        newPhotos[index] = photo.uri;
        setPhotos(newPhotos);
        
        if (photo.base64) {
          setImageBase64(photo.base64);
          console.log('✅ Base64 set from camera, length:', photo.base64.length);
        }
        
        setCameraVisible(false);
      }
    } catch (error) {
      console.error('❌ Error taking photo:', error);
      Alert.alert('Error', 'Failed to take photo');
    }
  };

  const pickImage = async (index) => {
    try {
      console.log('🖼️ Picking image for index:', index);
      
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      console.log('📱 ImagePicker result:', result);

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const imageUri = result.assets[0].uri;
        console.log('✅ Image selected:', imageUri);
        
        const newPhotos = [...photos];
        newPhotos[index] = imageUri;
        setPhotos(newPhotos);

        // Convert to base64
        const base64 = await convertImageToBase64(imageUri);
        setImageBase64(base64);
      }
    } catch (error) {
      console.error('❌ Error picking image:', error);
      Alert.alert('Error', 'Failed to pick image');
    }
  };

  const registerStudent = async () => {
    try {
      console.log('🚀 Starting registration...');
      
      // Validation
      if (!studentId.trim()) {
        Alert.alert('Error', 'Please enter Student ID');
        return;
      }
      
      if (!studentName.trim()) {
        Alert.alert('Error', 'Please enter Student Name');
        return;
      }
      
      if (!studentEmail.trim()) {
        Alert.alert('Error', 'Please enter Student Email');
        return;
      }
      
      // Check if we have at least 3 photos
      const validPhotos = photos.filter(photo => photo !== null);
      if (validPhotos.length < 3) {
        Alert.alert('Error', 'Please take all 3 photos');
        return;
      }
      
      setLoading(true);
      
      // Convert all 3 photos to base64
      console.log('🔄 Converting photos to base64...');
      const base64Images = [];
      
      for (let i = 0; i < 3; i++) {
        if (photos[i]) {
          const base64Image = await convertImageToBase64(photos[i]);
          base64Images.push(base64Image);
        } else {
          Alert.alert('Error', `Photo ${i + 1} is missing`);
          setLoading(false);
          return;
        }
      }
      
      // Prepare request body with all 3 images
      const requestBody = {
        class_code: classData.class_code,
        student_id: studentId.trim(),
        image1: base64Images[0],
        image2: base64Images[1],
        image3: base64Images[2]
      };
      
      console.log('📋 Request body prepared:', {
        class_code: requestBody.class_code,
        student_id: requestBody.student_id,
        image1Length: requestBody.image1.length,
        image2Length: requestBody.image2.length,
        image3Length: requestBody.image3.length
      });
      
      // Make API call
      const response = await fetch('https://7f21-14-186-25-188.ngrok-free.app/insert_student', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify(requestBody),
      });
      
      const result = await response.json();
      console.log('📥 Server response:', result);
      
      if (response.ok) {
        Alert.alert('Success', 'Student registered successfully!');
        // Reset form
        setStudentId('');
        setStudentName('');
        setStudentEmail('');
        setPhotos([null, null, null]);
        setImageBase64('');
      } else {
        Alert.alert('Error', result.message || result.detail || 'Registration failed');
      }
    } catch (error) {
      console.error('❌ Registration error:', error);
      Alert.alert('Error', 'Failed to register student');
    } finally {
      setLoading(false);
    }
  };

  const openCamera = (index) => {
    console.log('📷 Opening camera for index:', index);
    setCurrentPhotoIndex(index);
    setCameraVisible(true);
  };

  const PhotoSlot = ({ index, photo }) => (
    <View style={styles.photoSlot}>
      {photo ? (
        <Image source={{ uri: photo }} style={styles.photoPreview} />
      ) : (
        <View style={styles.emptyPhoto}>
          <Text style={styles.emptyPhotoText}>Photo {index + 1}</Text>
        </View>
      )}
      <View style={styles.photoButtons}>
        <TouchableOpacity
          style={styles.photoButton}
          onPress={() => openCamera(index)}
        >
          <Text style={styles.photoButtonText}>📷</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.photoButton}
          onPress={() => pickImage(index)}
        >
          <Text style={styles.photoButtonText}>🖼️</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  if (cameraVisible) {
    return (
      <View style={styles.cameraContainer}>
        <Camera
          style={styles.camera}
          type={CameraType.front}
          ref={cameraRef}
        >
          <View style={styles.cameraButtonContainer}>
            <TouchableOpacity
              style={styles.captureButton}
              onPress={() => takePhoto(currentPhotoIndex)}
            >
              <Text style={styles.captureButtonText}>Capture</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.cancelButton}
              onPress={() => setCameraVisible(false)}
            >
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </Camera>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <Card style={styles.card}>
          <Card.Content>
            <Title style={styles.title}>Register Student</Title>
            <Text style={styles.classInfo}>Class: {classData.class_name}</Text>
            
            <TextInput
              style={styles.input}
              placeholder="Student ID"
              value={studentId}
              onChangeText={setStudentId}
            />
            
            <TextInput
              style={styles.input}
              placeholder="Student Name"
              value={studentName}
              onChangeText={setStudentName}
            />
            
            <TextInput
              style={styles.input}
              placeholder="Student Email"
              value={studentEmail}
              onChangeText={setStudentEmail}
            />

            <Text style={styles.sectionTitle}>Photos</Text>
            <View style={styles.photosContainer}>
              {photos.map((photo, index) => (
                <PhotoSlot key={index} index={index} photo={photo} />
              ))}
            </View>

            <Button
              mode="contained"
              onPress={registerStudent}
              loading={loading}
              disabled={loading || !studentId.trim() || !imageBase64}
              style={styles.registerButton}
            >
              Register Student
            </Button>
          </Card.Content>
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContainer: {
    padding: 16,
  },
  card: {
    marginBottom: 16,
  },
  title: {
    textAlign: 'center',
    marginBottom: 16,
  },
  classInfo: {
    textAlign: 'center',
    marginBottom: 20,
    fontSize: 16,
    color: '#666',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    padding: 12,
    marginBottom: 16,
    borderRadius: 8,
    backgroundColor: 'white',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  photosContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  photoSlot: {
    width: '30%',
    aspectRatio: 0.75,
    marginHorizontal: '1.5%',
    marginBottom: 15,
  },
  photoPreview: {
    width: '100%',
    flex: 1,
    borderRadius: 8,
    marginBottom: 8,
  },
  emptyPhoto: {
    flex: 1,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#ddd',
    borderStyle: 'dashed',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  emptyPhotoText: {
    color: '#666',
    fontSize: 12,
    fontWeight: '500',
  },
  photoButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 4,
  },
  photoButton: {
    flex: 1,
    backgroundColor: '#007AFF',
    paddingVertical: 6,
    paddingHorizontal: 4,
    borderRadius: 6,
    alignItems: 'center',
  },
  photoButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  registerButton: {
    marginTop: 20,
  },
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  cameraButtonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'flex-end',
    paddingBottom: 50,
  },
  captureButton: {
    backgroundColor: '#007AFF',
    padding: 20,
    borderRadius: 50,
    alignItems: 'center',
  },
  captureButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  cancelButton: {
    backgroundColor: '#FF3B30',
    padding: 20,
    borderRadius: 50,
    alignItems: 'center',
  },
  cancelButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
}); 
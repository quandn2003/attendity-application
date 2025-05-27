// components/ClassDetails.js
import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  Alert,
  Image
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function ClassDetails({ route, navigation }) {
  const { classData } = route.params;
  const [students, setStudents] = useState([]);

  useEffect(() => {
    fetchStudents();
  }, []);

  const fetchStudents = async () => {
    try {
      // Mock data - replace with actual API call
      const mockStudents = [
        {
          id: '1',
          student_id: 'ST001',
          name: 'John Doe',
          is_present: true,
          attendance_time: '09:15 AM'
        },
        {
          id: '2',
          student_id: 'ST002',
          name: 'Jane Smith',
          is_present: false
        },
        {
          id: '3',
          student_id: 'ST003',
          name: 'Mike Johnson',
          is_present: true,
          attendance_time: '09:12 AM'
        }
      ];
      setStudents(mockStudents);
    } catch (error) {
      Alert.alert('Error', 'Failed to fetch students');
    }
  };

  const handleTakeAttendance = () => {
    navigation.navigate('AttendanceScanner', { classData });
  };

  const presentStudents = students.filter(s => s.is_present);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <View style={styles.headerInfo}>
          <Text style={styles.className}>{classData.class_name}</Text>
          <Text style={styles.classCode}>Code: {classData.class_code}</Text>
        </View>
      </View>

      <ScrollView style={styles.content}>
        {/* Stats Cards */}
        <View style={styles.statsContainer}>
          <View style={styles.statCard}>
            <Text style={styles.statNumber}>{students.length}</Text>
            <Text style={styles.statLabel}>Total Students</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={[styles.statNumber, { color: '#27ae60' }]}>
              {presentStudents.length}
            </Text>
            <Text style={styles.statLabel}>Present Today</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={[styles.statNumber, { color: '#3498db' }]}>
              {Math.round((presentStudents.length / students.length) * 100)}%
            </Text>
            <Text style={styles.statLabel}>Attendance Rate</Text>
          </View>
        </View>

        {/* Take Attendance Button */}
        <TouchableOpacity
          style={styles.attendanceButton}
          onPress={handleTakeAttendance}
        >
          <Text style={styles.attendanceButtonText}>📷 Take Attendance</Text>
        </TouchableOpacity>

        {/* Students List */}
        <View style={styles.studentsSection}>
          <Text style={styles.sectionTitle}>Students</Text>
          {students.map((student) => (
            <View key={student.id} style={styles.studentCard}>
              <View style={styles.studentInfo}>
                <View style={styles.avatar}>
                  <Text style={styles.avatarText}>
                    {student.name.split(' ').map(n => n[0]).join('')}
                  </Text>
                </View>
                <View style={styles.studentDetails}>
                  <Text style={styles.studentName}>{student.name}</Text>
                  <Text style={styles.studentId}>ID: {student.student_id}</Text>
                </View>
              </View>
              
              <View style={styles.attendanceStatus}>
                {student.is_present ? (
                  <View>
                    <View style={styles.presentBadge}>
                      <Text style={styles.presentText}>Present</Text>
                    </View>
                    {student.attendance_time && (
                      <Text style={styles.timeText}>{student.attendance_time}</Text>
                    )}
                  </View>
                ) : (
                  <View style={styles.absentBadge}>
                    <Text style={styles.absentText}>Absent</Text>
                  </View>
                )}
              </View>
            </View>
          ))}
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
  },
  backButton: {
    marginBottom: 10,
  },
  backButtonText: {
    color: '#3498db',
    fontSize: 16,
  },
  headerInfo: {
    alignItems: 'center',
  },
  className: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  classCode: {
    fontSize: 14,
    color: '#7f8c8d',
    marginTop: 5,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  statCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    alignItems: 'center',
    flex: 1,
    marginHorizontal: 5,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  statLabel: {
    fontSize: 12,
    color: '#7f8c8d',
    marginTop: 5,
    textAlign: 'center',
  },
  attendanceButton: {
    backgroundColor: '#3498db',
    borderRadius: 12,
    padding: 15,
    alignItems: 'center',
    marginBottom: 20,
    shadowColor: '#3498db',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 4.65,
    elevation: 8,
  },
  attendanceButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  studentsSection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 15,
  },
  studentCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f2f6',
  },
  studentInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#3498db',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  avatarText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
  },
  studentDetails: {
    flex: 1,
  },
  studentName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
  },
  studentId: {
    fontSize: 12,
    color: '#7f8c8d',
    marginTop: 2,
  },
  attendanceStatus: {
    alignItems: 'flex-end',
  },
  presentBadge: {
    backgroundColor: '#e8f5e8',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  presentText: {
    color: '#27ae60',
    fontSize: 12,
    fontWeight: 'bold',
  },
  absentBadge: {
    backgroundColor: '#ffeaea',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  absentText: {
    color: '#e74c3c',
    fontSize: 12,
    fontWeight: 'bold',
  },
  timeText: {
    fontSize: 10,
    color: '#7f8c8d',
    marginTop: 2,
  },
});
// components/ClassDetails.js
import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import ApiService from '../services/apiService';

export default function ClassDetails({ route, navigation }) {
  const { classData } = route.params;
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [classStats, setClassStats] = useState(null);

  useEffect(() => {
    fetchClassData();
  }, []);

  const fetchClassData = async () => {
    try {
      setLoading(true);
      console.log('=== Fetching class data for:', classData.class_code);
      
      const statsResponse = await ApiService.getClassStats(classData.class_code);
      console.log('=== Class stats response ===');
      console.log(JSON.stringify(statsResponse, null, 2));
      console.log('=== End class stats ===');
      
      setClassStats(statsResponse);
      
      // Extract students from the stats response
      if (statsResponse && statsResponse.students && statsResponse.students.length > 0) {
        const studentsWithAttendance = statsResponse.students.map((student, index) => ({
          id: index.toString(),
          student_id: student.student_id,
          name: student.name || student.student_id,
          email: student.email || 'No email',
          is_present: student.is_present || false,
          attendance_time: student.attendance_time || null,
          attendance_rate: student.attendance_rate || 0
        }));
        setStudents(studentsWithAttendance);
      } else {
        console.log('No students found in API response - students array is empty');
        // Create placeholder students based on recent attendance
        const uniqueStudentIds = [...new Set(statsResponse.recent_attendance?.map(r => r.student_id) || [])];
        const placeholderStudents = uniqueStudentIds.map((studentId, index) => {
          const latestAttendance = statsResponse.recent_attendance
            ?.filter(r => r.student_id === studentId)
            ?.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
          
          const today = new Date().toDateString();
          const attendanceToday = new Date(latestAttendance?.timestamp || '').toDateString();
          const isPresentToday = attendanceToday === today && latestAttendance?.status === 'present';
          
          return {
            id: index.toString(),
            student_id: studentId,
            name: studentId, // Use student_id as name since we don't have real names
            email: 'No email',
            is_present: isPresentToday,
            attendance_time: isPresentToday ? new Date(latestAttendance.timestamp).toLocaleTimeString() : null,
            attendance_rate: 85 // Default rate since we can't calculate it
          };
        });
        
        setStudents(placeholderStudents);
        
        // Update classStats to reflect the actual data
        setClassStats(prev => ({
          ...prev,
          total_students: placeholderStudents.length || statsResponse.student_count || 0,
          present_today: placeholderStudents.filter(s => s.is_present).length || statsResponse.present_today || 0
        }));
      }
      
    } catch (error) {
      console.error('Error fetching class data:', error);
      Alert.alert('Error', 'Failed to fetch class details. Please check your connection.');
      
      // Set empty state on error
      setStudents([]);
      setClassStats({
        total_students: 0,
        present_today: 0,
        attendance_rate: 0
      });
    } finally {
      setLoading(false);
    }
  };

  // Mock students data for testing
  const getMockStudents = () => {
    return [
      {
        id: '1',
        student_id: 'STU001',
        name: 'John Doe',
        email: 'john.doe@example.com',
        is_present: true,
        attendance_time: '09:15 AM',
        attendance_rate: 85
      },
      {
        id: '2',
        student_id: 'STU002',
        name: 'Jane Smith',
        email: 'jane.smith@example.com',
        is_present: true,
        attendance_time: '09:20 AM',
        attendance_rate: 92
      },
      {
        id: '3',
        student_id: 'STU003',
        name: 'Bob Wilson',
        email: 'bob.wilson@example.com',
        is_present: false,
        attendance_time: null,
        attendance_rate: 78
      },
      {
        id: '4',
        student_id: 'STU004',
        name: 'Alice Brown',
        email: 'alice.brown@example.com',
        is_present: false,
        attendance_time: null,
        attendance_rate: 65
      },
      {
        id: '5',
        student_id: 'STU005',
        name: 'Charlie Davis',
        email: 'charlie.davis@example.com',
        is_present: false,
        attendance_time: null,
        attendance_rate: 88
      }
    ];
  };

  const handleTakeAttendance = () => {
    navigation.navigate('AttendanceScanner', { classData });
  };

  const presentStudents = students.filter(s => s.is_present);
  const totalStudents = students.length;
  const attendanceRate = totalStudents > 0 ? Math.round((presentStudents.length / totalStudents) * 100) : 0;

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#3498db" />
          <Text style={styles.loadingText}>Loading class details...</Text>
        </View>
      </SafeAreaView>
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
        <View style={styles.headerInfo}>
          <Text style={styles.className}>{classData.class_name}</Text>
          <Text style={styles.classCode}>Code: {classData.class_code}</Text>
        </View>
      </View>

      <ScrollView style={styles.content}>
        {/* Stats Cards */}
        <View style={styles.statsContainer}>
          <View style={styles.statCard}>
            <Text style={styles.statNumber}>
              {classStats?.student_count || 0}
            </Text>
            <Text style={styles.statLabel}>Total Students</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statNumber}>
              {classStats?.present_today || 0}
            </Text>
            <Text style={styles.statLabel}>Present Today</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statNumber}>
              {Math.round(classStats?.attendance_rate || 0)}%
            </Text>
            <Text style={styles.statLabel}>Attendance Rate</Text>
          </View>
        </View>

        {/* Take Attendance Button */}
        <TouchableOpacity style={styles.attendanceButton} onPress={handleTakeAttendance}>
          <Text style={styles.attendanceButtonText}>📸 Take Attendance</Text>
        </TouchableOpacity>

        {/* Add Student Button */}
        <TouchableOpacity 
          style={[styles.attendanceButton, { backgroundColor: '#27ae60', marginTop: 10 }]} 
          onPress={() => navigation.navigate('StudentRegistration', { classData })}
        >
          <Text style={styles.attendanceButtonText}>👤 Add New Student</Text>
        </TouchableOpacity>

        {/* Students List */}
        <View style={styles.studentsSection}>
          <Text style={styles.sectionTitle}>Students ({totalStudents})</Text>
          
          {students.length === 0 ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyStateText}>No students enrolled</Text>
              <Text style={styles.emptyStateSubtext}>
                Students will appear here once they are registered for this class
              </Text>
            </View>
          ) : (
            students.map((student) => (
              <View key={student.id} style={styles.studentCard}>
                <View style={styles.studentInfo}>
                  <Text style={styles.studentName}>{student.name}</Text>
                  <Text style={styles.studentId}>ID: {student.student_id}</Text>
                  {student.email && student.email !== 'No email' && (
                    <Text style={styles.studentEmail}>{student.email}</Text>
                  )}
                  <Text style={styles.attendanceRate}>
                    Attendance Rate: {student.attendance_rate}%
                  </Text>
                </View>
                <View style={styles.attendanceStatus}>
                  <View style={[
                    styles.statusIndicator,
                    { backgroundColor: student.is_present ? '#27ae60' : '#e74c3c' }
                  ]}>
                    <Text style={styles.statusText}>
                      {student.is_present ? '✓' : '✗'}
                    </Text>
                  </View>
                  {student.attendance_time && (
                    <Text style={styles.attendanceTime}>{student.attendance_time}</Text>
                  )}
                </View>
              </View>
            ))
          )}
        </View>

        {/* Recent Activity */}
        {classStats?.recent_attendance && classStats.recent_attendance.length > 0 && (
          <View style={styles.activitySection}>
            <Text style={styles.sectionTitle}>Recent Activity</Text>
            {classStats.recent_attendance.slice(0, 5).map((record, index) => (
              <View key={index} style={styles.activityItem}>
                <Text style={styles.activityStudent}>{record.student_id}</Text>
                <Text style={styles.activityTime}>{record.timestamp}</Text>
                <View style={[
                  styles.activityStatus,
                  { backgroundColor: record.status === 'present' ? '#e8f5e8' : '#ffeaa7' }
                ]}>
                  <Text style={[
                    styles.activityStatusText,
                    { color: record.status === 'present' ? '#27ae60' : '#f39c12' }
                  ]}>
                    {record.status}
                  </Text>
                </View>
              </View>
            ))}
          </View>
        )}

        {/* Debug Info (remove in production) */}
        <View style={styles.debugSection}>
          <Text style={styles.debugTitle}>Debug Info:</Text>
          <Text style={styles.debugText}>Class Code: {classData.class_code}</Text>
          <Text style={styles.debugText}>Students Count: {students.length}</Text>
          <Text style={styles.debugText}>Present Count: {presentStudents.length}</Text>
          <Text style={styles.debugText}>API Stats: {classStats ? 'Loaded' : 'Failed'}</Text>
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#7f8c8d',
  },
  header: {
    backgroundColor: '#fff',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#e1e8ed',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  backButton: {
    padding: 5,
  },
  backButtonText: {
    fontSize: 16,
    color: '#3498db',
    fontWeight: '600',
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
    marginBottom: 20,
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
    flex: 1,
  },
  studentName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  studentId: {
    fontSize: 14,
    color: '#7f8c8d',
    marginTop: 2,
  },
  studentEmail: {
    fontSize: 12,
    color: '#95a5a6',
    marginTop: 2,
  },
  attendanceRate: {
    fontSize: 12,
    color: '#3498db',
    marginTop: 2,
  },
  attendanceStatus: {
    alignItems: 'center',
  },
  statusIndicator: {
    width: 30,
    height: 30,
    borderRadius: 15,
    justifyContent: 'center',
    alignItems: 'center',
  },
  statusText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  attendanceTime: {
    fontSize: 10,
    color: '#7f8c8d',
    marginTop: 5,
  },
  emptyState: {
    alignItems: 'center',
    padding: 20,
  },
  emptyStateText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#7f8c8d',
    marginBottom: 5,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#95a5a6',
    textAlign: 'center',
  },
  activitySection: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    marginBottom: 20,
  },
  activityItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f2f6',
  },
  activityStudent: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#2c3e50',
    flex: 1,
  },
  activityTime: {
    fontSize: 12,
    color: '#7f8c8d',
    flex: 1,
    textAlign: 'center',
  },
  activityStatus: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  activityStatusText: {
    fontSize: 12,
    fontWeight: 'bold',
    textTransform: 'uppercase',
  },
  debugSection: {
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 10,
    marginTop: 10,
    borderWidth: 1,
    borderColor: '#dee2e6',
  },
  debugTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#6c757d',
    marginBottom: 5,
  },
  debugText: {
    fontSize: 12,
    color: '#6c757d',
    marginBottom: 2,
  },
});
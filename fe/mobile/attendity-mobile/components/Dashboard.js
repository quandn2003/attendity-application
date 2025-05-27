// components/Dashboard.js
import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  Alert,
  RefreshControl
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function Dashboard({ navigation }) {
  const [classes, setClasses] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    fetchClasses();
  }, []);

  const fetchClasses = async () => {
    try {
      // Mock data - replace with actual API call
      const mockClasses = [
        {
          id: '1',
          class_code: 'CS101',
          class_name: 'Introduction to Computer Science',
          student_count: 25,
          present_today: 18,
          last_session: '2024-01-15'
        },
        {
          id: '2',
          class_code: 'MATH201',
          class_name: 'Advanced Mathematics',
          student_count: 30,
          present_today: 22,
          last_session: '2024-01-14'
        }
      ];
      setClasses(mockClasses);
    } catch (error) {
      Alert.alert('Error', 'Failed to fetch classes');
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchClasses();
    setRefreshing(false);
  };

  const handleClassPress = (classItem) => {
    navigation.navigate('ClassDetails', { classData: classItem });
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>My Classes</Text>
        <Text style={styles.subtitle}>Manage your class attendance</Text>
      </View>

      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {classes.map((classItem) => (
          <TouchableOpacity
            key={classItem.id}
            style={styles.classCard}
            onPress={() => handleClassPress(classItem)}
          >
            <View style={styles.classHeader}>
              <Text style={styles.classCode}>{classItem.class_code}</Text>
              <View style={styles.attendanceRate}>
                <Text style={styles.attendanceText}>
                  {Math.round((classItem.present_today / classItem.student_count) * 100)}%
                </Text>
              </View>
            </View>
            
            <Text style={styles.className}>{classItem.class_name}</Text>
            
            <View style={styles.classStats}>
              <View style={styles.stat}>
                <Text style={styles.statNumber}>{classItem.student_count}</Text>
                <Text style={styles.statLabel}>Total Students</Text>
              </View>
              <View style={styles.stat}>
                <Text style={styles.statNumber}>{classItem.present_today}</Text>
                <Text style={styles.statLabel}>Present Today</Text>
              </View>
            </View>
          </TouchableOpacity>
        ))}
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
    padding: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
    color: '#7f8c8d',
  },
  scrollView: {
    flex: 1,
    padding: 20,
  },
  classCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  classHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  classCode: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#3498db',
  },
  attendanceRate: {
    backgroundColor: '#e8f5e8',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  attendanceText: {
    color: '#27ae60',
    fontWeight: 'bold',
    fontSize: 14,
  },
  className: {
    fontSize: 16,
    color: '#2c3e50',
    marginBottom: 15,
  },
  classStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  stat: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  statLabel: {
    fontSize: 12,
    color: '#7f8c8d',
    marginTop: 2,
  },
});
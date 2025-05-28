// components/Dashboard.js
import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  Alert,
  RefreshControl,
  ActivityIndicator
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import ApiService from '../services/apiService';

export default function Dashboard({ navigation }) {
  const [classes, setClasses] = useState([]);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchClasses();
  }, []);

  const fetchClasses = async () => {
    try {
      setLoading(true);
      const response = await ApiService.getAllClasses();
      
      if (response.status === 'success') {
        // Use real data from API - no more mocking
        const transformedClasses = response.classes.map((classItem, index) => ({
          id: index.toString(),
          class_code: classItem.class_code,
          class_name: classItem.class_name || `${classItem.class_code} Course`,
          student_count: classItem.student_count || 0,
          present_today: classItem.present_today || 0,
          last_session: classItem.last_session || classItem.created_at || 'No sessions yet',
          status: classItem.status || 'active',
          attendance_rate: classItem.attendance_rate || 0
        }));
        setClasses(transformedClasses);
      } else {
        setClasses([]);
      }
    } catch (error) {
      console.error('Error fetching classes:', error);
      Alert.alert('Error', 'Failed to fetch classes. Please check your connection.');
      setClasses([]);
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchClasses();
    setRefreshing(false);
  };

  const handleClassPress = async (classItem) => {
    try {
      // Fetch detailed class stats before navigating
      const statsResponse = await ApiService.getClassStats(classItem.class_code);
      
      const enhancedClassData = {
        ...classItem,
        ...statsResponse,
        detailed_stats: statsResponse
      };
      
      navigation.navigate('ClassDetails', { classData: enhancedClassData });
    } catch (error) {
      console.error('Error fetching class details:', error);
      // Navigate with basic data if stats fetch fails
      navigation.navigate('ClassDetails', { classData: classItem });
    }
  };

  const handleCreateClass = () => {
    Alert.prompt(
      'Create New Class',
      'Enter class code:',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Create',
          onPress: async (classCode) => {
            if (classCode && classCode.trim()) {
              try {
                await ApiService.createClass(classCode.trim());
                Alert.alert('Success', `Class ${classCode} created successfully!`);
                fetchClasses(); // Refresh the list
              } catch (error) {
                Alert.alert('Error', 'Failed to create class. Please try again.');
              }
            }
          }
        }
      ],
      'plain-text'
    );
  };

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#3498db" />
          <Text style={styles.loadingText}>Loading classes...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <Text style={styles.title}>My Classes</Text>
          <Text style={styles.subtitle}>Manage your class attendance</Text>
        </View>
        <TouchableOpacity style={styles.createButton} onPress={handleCreateClass}>
          <Text style={styles.createButtonText}>+ New Class</Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {classes.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyStateText}>No classes found</Text>
            <Text style={styles.emptyStateSubtext}>
              Create your first class to get started
            </Text>
            <TouchableOpacity style={styles.emptyCreateButton} onPress={handleCreateClass}>
              <Text style={styles.emptyCreateButtonText}>Create Class</Text>
            </TouchableOpacity>
          </View>
        ) : (
          classes.map((classItem) => (
            <TouchableOpacity
              key={classItem.id}
              style={styles.classCard}
              onPress={() => handleClassPress(classItem)}
            >
              <View style={styles.classHeader}>
                <Text style={styles.classCode}>{classItem.class_code}</Text>
                <View style={[
                  styles.statusBadge,
                  { backgroundColor: classItem.status === 'active' ? '#e8f5e8' : '#ffeaa7' }
                ]}>
                  <Text style={[
                    styles.statusText,
                    { color: classItem.status === 'active' ? '#27ae60' : '#f39c12' }
                  ]}>
                    {classItem.status}
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
                <View style={styles.stat}>
                  <Text style={styles.statNumber}>{classItem.attendance_rate}%</Text>
                  <Text style={styles.statLabel}>Attendance Rate</Text>
                </View>
              </View>
            </TouchableOpacity>
          ))
        )}
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
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
  },
  headerContent: {
    flex: 1,
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
  createButton: {
    backgroundColor: '#27ae60',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 8,
  },
  createButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
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
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  statusText: {
    fontWeight: 'bold',
    fontSize: 12,
    textTransform: 'uppercase',
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
  emptyState: {
    alignItems: 'center',
    padding: 40,
    marginTop: 50,
  },
  emptyStateText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#7f8c8d',
    marginBottom: 8,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#95a5a6',
    textAlign: 'center',
    marginBottom: 20,
  },
  emptyCreateButton: {
    backgroundColor: '#3498db',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
  },
  emptyCreateButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
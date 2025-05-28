import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import LoginScreen from './LoginScreen'; // Your current login component
import Dashboard from './components/Dashboard';
import ClassDetails from './components/ClassDetails';
import AttendanceScanner from './components/AttendanceScanner';
import StudentRegistration from './components/StudentRegistration';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator 
        initialRouteName="Login"
        screenOptions={{ headerShown: false }}
      >
        <Stack.Screen name="Login" component={LoginScreen} />
        <Stack.Screen name="Dashboard" component={Dashboard} />
        <Stack.Screen name="ClassDetails" component={ClassDetails} />
        <Stack.Screen name="AttendanceScanner" component={AttendanceScanner} />
        <Stack.Screen 
          name="StudentRegistration" 
          component={StudentRegistration}
          options={{ headerShown: false }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
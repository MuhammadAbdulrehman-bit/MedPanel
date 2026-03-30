import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Image, ScrollView, ActivityIndicator, TextInput } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export default function App() {
  const [image, setImage] = useState(null);
  const [patientName, setPatientName] = useState("");
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);

  // The missing function that was causing your error!
  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const handleAnalyze = async () => {
    if (!image || !patientName) {
      alert("Please enter a patient name and select an image.");
      return;
    }
    
    setLoading(true);
    try {
      const formData = new FormData();
      const response = await fetch(image);
      const blob = await response.blob();
      
      formData.append('file', blob, 'scan.jpg');
      formData.append('patient_name', patientName);

      // Use your direct Hugging Face URL
      const apiResponse = await fetch("https://fhkhthjkkl-medpanel.hf.space/analyze", {
        method: 'POST',
        body: formData,
      });

      const data = await apiResponse.json();
      setReport(data);
    } catch (error) {
      console.error(error);
      alert("Analysis failed. Check if Hugging Face is awake!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={{ flex: 1, backgroundColor: '#0f172a', padding: 20 }}>
      <View style={{ marginTop: 50, alignItems: 'center' }}>
        <Text style={{ color: '#10b981', fontSize: 32, fontWeight: 'bold' }}>MediScan AI</Text>
        <Text style={{ color: '#94a3b8', marginTop: 5 }}>Diagnostic Triage Interface</Text>
      </View>

      <View style={{ mt: 30 }}>
        <TextInput 
          style={{ backgroundColor: '#1e293b', color: 'white', padding: 15, borderRadius: 12, marginTop: 30, borderWeight: 1, borderColor: '#334155' }}
          placeholder="Enter Patient Name..."
          placeholderTextColor="#64748b"
          value={patientName}
          onChangeText={setPatientName}
        />

        <TouchableOpacity 
          onPress={pickImage} 
          style={{ height: 250, backgroundColor: '#1e293b', marginTop: 20, borderRadius: 20, justifyContent: 'center', alignItems: 'center', borderStyle: 'dashed', borderWidth: 2, borderColor: '#475569', overflow: 'hidden' }}
        >
          {image ? (
            <Image source={{ uri: image }} style={{ width: '100%', height: '100%' }} />
          ) : (
            <Text style={{ color: '#94a3b8' }}>Click to Upload Medical Scan</Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity 
          onPress={handleAnalyze}
          disabled={loading}
          style={{ backgroundColor: loading ? '#334155' : '#059669', padding: 18, borderRadius: 12, marginTop: 25, alignItems: 'center' }}
        >
          {loading ? <ActivityIndicator color="white" /> : <Text style={{ color: 'white', fontWeight: 'bold', fontSize: 16 }}>RUN AI DIAGNOSIS</Text>}
        </TouchableOpacity>
      </View>

      {report && (
        <View style={{ marginTop: 40, padding: 20, backgroundColor: '#1e293b', borderRadius: 20, borderLeftWidth: 5, borderLeftColor: '#10b981', marginBottom: 50 }}>
          <Text style={{ color: '#10b981', fontWeight: 'bold', fontSize: 18 }}>Analysis Report:</Text>
          <Text style={{ color: '#fbbf24', fontWeight: 'bold', marginTop: 10 }}>Finding: {report.top_finding}</Text>
          <Text style={{ color: '#e2e8f0', marginTop: 10, lineHeight: 22 }}>{report.report}</Text>
        </View>
      )}
    </ScrollView>
  );
}
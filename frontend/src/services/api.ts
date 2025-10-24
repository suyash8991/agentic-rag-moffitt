import axios from 'axios';

// Constants
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
const API_KEY = process.env.REACT_APP_API_KEY || 'dev_api_key';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
  },
});

// Types
export interface QueryRequest {
  query: string;
  streaming?: boolean;
  max_results?: number;
}

export interface QueryResponse {
  query_id: string;
  query: string;
  answer: string;
  completed: boolean;
  error?: string;
}

// API functions
export const fetchHealth = async (): Promise<{ status: string }> => {
  const response = await api.get('/health');
  return response.data;
};

export const fetchSettings = async () => {
  const response = await api.get('/settings');
  return response.data;
};

export const fetchDepartments = async (): Promise<string[]> => {
  const response = await api.get('/departments');
  return response.data.departments;
};

export const fetchPrograms = async (): Promise<string[]> => {
  const response = await api.get('/programs');
  return response.data.programs;
};

export const sendQuery = async (query: string): Promise<QueryResponse> => {
  const requestBody: QueryRequest = {
    query,
    streaming: false,
    max_results: 5,
  };

  const response = await api.post('/query', requestBody);
  return response.data;
};

export default api;
import React, { useState, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { sendQuery, fetchHealth } from '../../services/api';

interface Message {
  content: string;
  isUser: boolean;
}

const ChatContainer: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      content: 'Hello! I\'m the Moffitt Researcher Assistant. How can I help you today?',
      isUser: false
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [isBackendAvailable, setIsBackendAvailable] = useState(true);

  // Check if the backend is available when component mounts
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        await fetchHealth();
        setIsBackendAvailable(true);
      } catch (error) {
        console.error('Backend unavailable:', error);
        setIsBackendAvailable(false);
        setMessages(prevMessages => [
          ...prevMessages,
          {
            content: 'Warning: Unable to connect to the backend API. The chat functionality may not work properly.',
            isUser: false
          }
        ]);
      }
    };

    checkBackendStatus();
  }, []);

  const sendMessage = async (messageText: string) => {
    // Add user message to chat
    const userMessage: Message = {
      content: messageText,
      isUser: true
    };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setIsLoading(true);

    // Check if backend is available
    if (!isBackendAvailable) {
      const errorMessage: Message = {
        content: 'Cannot process your request. The backend API is currently unavailable. Please try again later.',
        isUser: false
      };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
      setIsLoading(false);
      return;
    }

    try {
      // Make the actual API call to send the query
      const response = await sendQuery(messageText);

      const botResponse: Message = {
        content: response.answer,
        isUser: false
      };
      setMessages(prevMessages => [...prevMessages, botResponse]);
      setIsLoading(false);
    } catch (error) {
      console.error('Error sending message:', error);
      setIsBackendAvailable(false); // Mark backend as unavailable if request fails

      const errorMessage: Message = {
        content: 'Sorry, there was an error processing your request. The backend may be unavailable. Please try again later.',
        isUser: false
      };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((message, index) => (
          <ChatMessage
            key={index}
            content={message.content}
            isUser={message.isUser}
          />
        ))}
      </div>
      <ChatInput onSendMessage={sendMessage} isLoading={isLoading} />
    </div>
  );
};

export default ChatContainer;
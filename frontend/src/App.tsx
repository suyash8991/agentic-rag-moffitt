import React from 'react';
import './App.css';
import ChatContainer from './components/Chat/ChatContainer';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Moffitt Researcher Assistant</h1>
        <p>Ask questions about Moffitt Cancer Center researchers</p>
      </header>
      <main className="App-main">
        <ChatContainer />
      </main>
      <footer className="App-footer">
        <p>© 2025 Moffitt Cancer Center</p>
      </footer>
    </div>
  );
}

export default App;

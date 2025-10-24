import './App.css';
import ChatContainer from './components/Chat/ChatContainer';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <div className="logo-container">
            <img src="/moffitt-logo-full.svg" alt="Moffitt Cancer Center" className="moffitt-logo" />
          </div>
          <div className="title-container">
            <h1>Moffitt Researcher Agent Chat Assistant</h1>
            <p>Ask questions about Moffitt Cancer Center researchers</p>
          </div>
        </div>
      </header>
      <main className="App-main">
        <ChatContainer />
      </main>
    </div>
  );
}

export default App;

import './App.css';
import ChatContainer from './components/Chat/ChatContainer';
// @ts-ignore
import moffittLogo from './assets/moffitt-logo-dark.svg';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <div className="logo-container">
            <img src={moffittLogo} alt="Moffitt Cancer Center" className="moffitt-logo" />
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
      <footer className="App-footer">
        <p>© 2025 Moffitt Cancer Center</p>
      </footer>
    </div>
  );
}

export default App;

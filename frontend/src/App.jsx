import { useState } from 'react'
// import fileupload from "./fileupload";
import fileupload from './fileupload';

import tldrLogo from './assets/tldresearch-logo.png'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div className="App">
        <h1 className="upload-title">Upload File for Summarization</h1>
        <fileupload/>
      </div>

    </>
  );
}

export default App




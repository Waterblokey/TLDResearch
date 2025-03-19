import React, { useState, useRef } from "react";

const fileupload = () => {
  const [file, setFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
    }
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div 
      className="upload-container"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      {file ? (
        <p>{file.name}</p>
      ) : (
        <p>Drag & drop a file here, or click to upload</p>
      )}
      <input 
        type="file" 
        ref={fileInputRef} 
        style={{ display: "none" }} 
        onChange={handleFileChange} 
        accept=".pdf,.txt,.docx"
      />
    </div>
  );
};

export default fileupload;

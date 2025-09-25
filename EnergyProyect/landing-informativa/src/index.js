import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
//import './tailwind.css';

//import 'bootstrap/dist/css/bootstrap.min.css'; // <--- importante
//import 'bootstrap-icons/font/bootstrap-icons.css';
//import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import reportWebVitals from './reportWebVitals';
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.getRegistrations().then((regs) => {
    regs.forEach((sw) => sw.unregister());
  });
}

//hata aqui

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);


// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

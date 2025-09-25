import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home.jsx"; // 👈 forzamos .jsx para evitar ambigüedades
import Login from "./pages/Login";
import Register from "./pages/Register";
import PrediccionDashboard from "./pages/PrediccionDashboard";
import GestionDashboard from "./pages/GestionDashboard";

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/home" element={<Home />} /> {/* opcional para pruebas */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/dashboard/prediccion" element={<PrediccionDashboard />} />
        <Route path="/dashboard/gestion" element={<GestionDashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

export default App;

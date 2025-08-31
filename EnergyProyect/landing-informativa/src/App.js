import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
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
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                <Route path="/dashboard/prediccion" element={<PrediccionDashboard />} />
                <Route path="/dashboard/gestion" element={<GestionDashboard />} />
            </Routes>
        </Router>
    );
}

export default App;

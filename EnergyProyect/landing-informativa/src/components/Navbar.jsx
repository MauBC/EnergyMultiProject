import { Link, useNavigate } from "react-router-dom";
import logo from "../assets/logo.png";

export default function Navbar() {
    const navigate = useNavigate();
    const token = localStorage.getItem("token");

    function handleLogout() {
        localStorage.removeItem("token");
        navigate("/");
    }

    return (
        <nav className="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
            <div className="container d-flex justify-content-between align-items-center">
                {/* Logo */}
                <div className="d-flex align-items-center">
                    <Link to="/">
                        <img src={logo} alt="Logo" height="40" style={{ marginLeft: '8px' }} />
                    </Link>
                </div>

                {/* Mobile menu button */}
                <button
                    className="navbar-toggler"
                    type="button"
                    data-bs-toggle="collapse"
                    data-bs-target="#navbarNav"
                >
                    <span className="navbar-toggler-icon"></span>
                </button>

                {/* Menu items */}
                <div className="collapse navbar-collapse justify-content-end" id="navbarNav">
                    <ul className="navbar-nav">
                        {/* Links a secciones del Home */}
                        <li className="nav-item">
                            <a className="nav-link" href="#soluciones">Soluciones</a>
                        </li>
                        <li className="nav-item">
                            <a className="nav-link" href="#resultados">Resultados</a>
                        </li>
                        <li className="nav-item">
                            <a className="nav-link" href="#nosotros">Nosotros</a>
                        </li>
                        <li className="nav-item">
                            <a className="nav-link" href="#contacto">Contáctanos</a>
                        </li>

                        {/* Botones de login/register o logout */}
                        {!token ? (
                            <>
                                <li className="nav-item">
                                    <Link className="nav-link" to="/login">Login</Link>
                                </li>
                                <li className="nav-item">
                                    <Link className="nav-link" to="/register">Register</Link>
                                </li>
                            </>
                        ) : (
                            <li className="nav-item">
                                <button
                                    className="btn btn-link nav-link"
                                    style={{ padding: 0 }}
                                    onClick={handleLogout}
                                >
                                    Cerrar sesión
                                </button>
                            </li>
                        )}
                    </ul>
                </div>
            </div>
        </nav>
    );
}

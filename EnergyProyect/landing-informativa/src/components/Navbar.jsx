import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const token = localStorage.getItem("token");

  const linkStyle = { textDecoration: "none" }; // 👈 fuerza sin subrayado
  const linkCls =
    "text-sm font-medium text-black hover:text-green-700 focus:text-green-700 transition-colors";

  const navClick = () => setOpen(false);

  const NavA = ({ href, children }) => (
    <a href={href} onClick={navClick} className={linkCls} style={linkStyle}>
      {children}
    </a>
  );

  const onLogout = () => {
    localStorage.removeItem("token");
    setOpen(false);
    navigate("/");
  };

  return (
    <header className="fixed top-0 inset-x-0 z-50 bg-white/95 backdrop-blur shadow">
      <div className="max-w-7xl mx-auto flex items-center justify-between h-16 px-4 md:px-8">
        {/* Marca: verde, sin subrayado */}
        <Link to="/" onClick={navClick} className="select-none" style={linkStyle}>
          <span className="text-2xl font-extrabold text-green-700">EnergIA</span>
        </Link>

        {/* Links desktop */}
        <nav className="hidden md:flex items-center gap-8">
          <NavA href="#soluciones">Soluciones</NavA>
          <NavA href="#resultados">Resultados</NavA>
          <NavA href="#nosotros">Nosotros</NavA>
          <NavA href="#contacto">Contáctanos</NavA>
        </nav>

        {/* Auth desktop */}
        <div className="hidden md:flex items-center gap-4">
          {!token ? (
            <>
              <Link to="/login" onClick={navClick} className={linkCls} style={linkStyle}>
                Login
              </Link>
              <Link to="/register" onClick={navClick} className={linkCls} style={linkStyle}>
                Register
              </Link>
            </>
          ) : (
            <button onClick={onLogout} className={linkCls} style={linkStyle}>
              Cerrar sesión
            </button>
          )}
        </div>

        {/* Botón menú móvil */}
        <button
          className="md:hidden inline-flex items-center justify-center rounded-md p-2 text-black hover:bg-gray-100"
          onClick={() => setOpen((v) => !v)}
          aria-label="Toggle menu"
        >
          <svg width="24" height="24" viewBox="0 0 24 24">
            <path
              fill="currentColor"
              d={open ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"}
            />
          </svg>
        </button>
      </div>

      {/* Menú móvil */}
      {open && (
        <div className="md:hidden border-t bg-white">
          <div className="px-4 py-3 flex flex-col gap-3">
            <NavA href="#soluciones">Soluciones</NavA>
            <NavA href="#resultados">Resultados</NavA>
            <NavA href="#nosotros">Nosotros</NavA>
            <NavA href="#contacto">Contáctanos</NavA>
            <div className="h-px bg-gray-200 my-1" />
            {!token ? (
              <>
                <Link to="/login" onClick={navClick} className={linkCls} style={linkStyle}>
                  Login
                </Link>
                <Link to="/register" onClick={navClick} className={linkCls} style={linkStyle}>
                  Register
                </Link>
              </>
            ) : (
              <button onClick={onLogout} className={linkCls + " text-left"} style={linkStyle}>
                Cerrar sesión
              </button>
            )}
          </div>
        </div>
      )}
    </header>
  );
}
        
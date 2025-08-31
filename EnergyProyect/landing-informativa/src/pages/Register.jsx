import { useState } from "react";
import { registerUser } from "../services/api";
import { useNavigate } from "react-router-dom";

export default function Register() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [company, setCompany] = useState("");
    const [role, setRole] = useState("user");
    const navigate = useNavigate();

    async function handleSubmit(e) {
        e.preventDefault();
        const data = await registerUser(email, password, company, role);
        if (data.msg) {
            alert("Usuario registrado con éxito");
            navigate("/login");
        } else {
            alert(data.detail || "Error en registro");
        }
    }

    return (
        <div className="container mt-5">
            <h2>Registro</h2>
            <form onSubmit={handleSubmit}>
                <input className="form-control mb-2" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
                <input type="password" className="form-control mb-2" placeholder="Contraseña" value={password} onChange={(e) => setPassword(e.target.value)} />
                <input className="form-control mb-2" placeholder="Empresa" value={company} onChange={(e) => setCompany(e.target.value)} />
                <select className="form-control mb-2" value={role} onChange={(e) => setRole(e.target.value)}>
                    <option value="user">Usuario</option>
                    <option value="admin">Administrador</option>
                </select>
                <button className="btn btn-success">Registrar</button>
            </form>
        </div>
    );
}

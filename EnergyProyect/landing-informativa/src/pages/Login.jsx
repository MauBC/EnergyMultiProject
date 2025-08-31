import { useState } from "react";
import { loginUser } from "../services/api";
import { useNavigate } from "react-router-dom";

export default function Login() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const navigate = useNavigate();

    async function handleSubmit(e) {
        e.preventDefault();
        const data = await loginUser(email, password);
        if (data.access_token) {
            localStorage.setItem("token", data.access_token);
            navigate("/");
        } else {
            alert(data.detail || "Error en login");
        }
    }

    return (
        <div className="container mt-5">
            <h2>Iniciar sesión</h2>
            <form onSubmit={handleSubmit}>
                <input className="form-control mb-2" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
                <input type="password" className="form-control mb-2" placeholder="Contraseña" value={password} onChange={(e) => setPassword(e.target.value)} />
                <button className="btn btn-primary">Entrar</button>
            </form>
        </div>
    );
}

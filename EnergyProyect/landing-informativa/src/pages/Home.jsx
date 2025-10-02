// src/pages/Home.jsx
import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  LineChart, // gráfico (recharts)
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Linkedin,
  Mail,
  Phone,
  CheckCircle2,
  BatteryCharging,
  LineChart as LineChartIcon, // alias del icono (evita conflicto con recharts)
  Leaf,
} from "lucide-react";

export default function Home() {
  const yearNow = new Date().getFullYear();

  // Datos del gráfico (demo)
  const data = [
    { year: "2025", co2: 50 },
    { year: "2026", co2: 150 },
    { year: "2027", co2: 300 },
  ];

  // Tarjetas de soluciones
  const soluciones = [
    {
      title: "Gestión Energética",
      desc: "Optimización con IA de consumos industriales.",
      img: "./gestion.png",
      beneficios: [
        "Reducción de costos operativos mediante la detección de consumos ocultos o ineficientes.",
        "Mantenimiento predictivo que disminuye fallos y paradas de equipos.",
        "Mejor control y visibilidad del consumo energético en tiempo real.",
        "Disminución de la huella de carbono y cumplimiento de metas de sostenibilidad.",
      ],
      referencia:
        "Khanum, M., Dahrouj, H., Bansal, R., & Tawfik, H. (2025). An Overview of the Prospects and Challenges of Using Artificial Intelligence for Energy Management Systems in Microgrids. arXiv.",
    },
    {
      title: "Gestión Energética de BESS",
      desc: "Modelos IA para gestionar BESS.",
      img: "./bess.png",
      beneficios: [
        "Optimización de los ciclos de carga y descarga, prolongando la vida útil de las baterías.",
        "Reducción de costos gracias a la gestión inteligente de tarifas horarias.",
        "Mayor integración de renovables al suavizar la intermitencia de la generación solar o eólica.",
        "Prestación de servicios auxiliares a la red (reserva, regulación de frecuencia, picos de demanda).",
      ],
      referencia:
        "Alkhayyat, H. et al. (2025). Optimisation of photovoltaic and battery systems for cost reduction in commercial buildings. Applied Energy, Elsevier.",
    },
    {
      title: "Predicción Fotovoltaica",
      desc: "Cálculo de energía solar según clima y ubicación.",
      img: "./fotovoltaica.png",
      beneficios: [
        "Predicciones más precisas de generación solar, facilitando la planificación energética.",
        "Mejor dimensionamiento de sistemas solares y de respaldo.",
        "Reducción de costos por desviaciones en la generación estimada.",
        "Mayor confiabilidad en la operación de la red eléctrica.",
      ],
      referencia:
        "Pillai, U. et al. (2023). Forecasting Methods for Photovoltaic Energy in the PV-BESS context. Energies, MDPI.",
    },
  ];

  const [openIdx, setOpenIdx] = useState(null);

  // arriba del return:
const [form, setForm] = useState({
  nombre: "",
  correo: "",
  empresa: "",
  mensaje: "",
  hp: "" // honeypot hidden
});
const [sending, setSending] = useState(false);
const [okMsg, setOkMsg] = useState("");
const [errMsg, setErrMsg] = useState("");

const API_URL = "https://tfc9kyrrtg.execute-api.us-east-1.amazonaws.com/contact";

function onChange(e) {
  setForm({ ...form, [e.target.name]: e.target.value });
}

async function onSubmit(e) {
  e.preventDefault();
  setOkMsg("");
  setErrMsg("");
  try {
    setSending(true);
    const r = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form)
    });
    const j = await r.json().catch(() => ({}));
    if (r.ok && j.ok) {
      setOkMsg("Mensaje enviado. Gracias!");
      setForm({ nombre: "", correo: "", empresa: "", mensaje: "", hp: "" });
    } else {
      setErrMsg(j.error || "No se pudo enviar. Intenta nuevamente.");
    }
  } catch (err) {
    setErrMsg("Error de red. Intenta nuevamente.");
  } finally {
    setSending(false);
  }
}


  return (
    <div className="font-sans">
      {/* HERO con imagen de fondo y CTA */}
      <section className="relative min-h-[80vh] md:min-h-screen pt-28 md:pt-36 px-6 flex flex-col items-center justify-center text-center text-white">
        {/* Fondo */}
        <img
          src="./imagen_fondo.png"
          alt=""
          className="absolute inset-0 -z-20 w-full h-full object-cover"
        />
        {/* Oscurecedor */}
        <div className="absolute inset-0 -z-10 bg-black/65" />

        {/* Título (tamaño medio entre el original y el gigante) */}
       <motion.h2
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.9 }}
        className="max-w-7xl mx-auto font-extrabold tracking-tight drop-shadow text-center 
                  text-3xl sm:text-5xl md:text-7xl lg:text-8xl leading-tight"
      >
        Optimizamos la energía con
        <br/>
        Inteligencia Artificial
      </motion.h2>


        <div className="mt-14 md:mt-16">
          <a
            href="#contacto"
            className="inline-flex rounded-2xl bg-green-600 px-6 py-3 text-white text-base md:text-lg font-semibold hover:bg-green-700"
          >
            Quiero ser socio
          </a>
        </div>
      </section>

      {/* Soluciones */}
      <section id="soluciones" className="scroll-mt-28 py-20 px-6 bg-gray-50 text-center">
        <h3 className="text-3xl font-bold mb-10">Soluciones</h3>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {soluciones.map((s, i) => {
            const isOpen = openIdx === i;
            return (
              <div key={i} className="shadow-lg rounded-2xl overflow-hidden bg-white text-left">
                <img src={s.img} alt={s.title} className="w-full h-40 object-cover" />

                <div className="p-6">
                  <h4 className="font-bold text-2xl mb-2">{s.title}</h4>
                  <p className="text-gray-600 mb-4">{s.desc}</p>

                  <button
                    type="button"
                    onClick={() => setOpenIdx(isOpen ? null : i)}
                    className="inline-flex items-center rounded-2xl bg-green-600 px-4 py-2 text-white font-semibold hover:bg-green-700"
                  >
                    {isOpen ? "Ocultar beneficios" : "Saber más"}
                  </button>

                  {isOpen && (
                    <div className="mt-5 space-y-3">
                      <h5 className="font-semibold text-gray-800">Beneficios</h5>
                      <ul className="list-disc pl-5 text-gray-700 space-y-1">
                        {s.beneficios.map((b, k) => (
                          <li key={k}>{b}</li>
                        ))}
                      </ul>
                      <p className="text-sm text-gray-500">
                        <span className="font-medium">Referencia: </span>
                        {s.referencia}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Resultados (solo gráfico centrado) */}
      <section id="resultados" className="scroll-mt-28 py-20 px-6 bg-white text-center">
        <h3 className="text-3xl font-bold mb-2">Resultados Proyectados</h3>
        <p className="mb-8 text-gray-600">
          Toneladas de CO₂ evitadas desde 2025 (proyección)
        </p>

        {/* Contenedor centrado del gráfico */}
        <div className="mx-auto max-w-4xl w-full h-[22rem] md:h-[28rem]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
            >
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="co2"
                stroke="#16a34a"
                strokeWidth={3}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Nosotros (versión pro) */}
      <section id="nosotros" className="scroll-mt-28 relative py-20 bg-gray-50">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(16,185,129,0.08),transparent_60%)]" />
        <div className="relative max-w-6xl mx-auto px-6">
          <div className="text-center mb-10">
            <span className="inline-block rounded-full bg-green-100 text-green-700 px-3 py-1 text-xs font-semibold tracking-wide">
              Quiénes somos
            </span>
            <h2 className="mt-3 text-3xl md:text-4xl font-bold text-gray-900">Nosotros</h2>
            <p className="mt-4 max-w-4xl mx-auto text-gray-600 text-lg leading-relaxed">
              EnergIA es una startup que utiliza inteligencia artificial (IA) para revolucionar la gestión energética.
              Implementamos soluciones que reducen consumo y emisiones de CO₂, a la vez que generan ahorros cuantificables.
              Acompañamos tus metas de Responsabilidad Social Empresarial (RSE) con herramientas como la gestión optimizada
              de baterías (BESS) y la predicción fotovoltaica para un futuro sostenible y rentable.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 items-stretch">
            <div className="bg-white rounded-2xl shadow p-6 md:p-8">
              <ul className="space-y-5">
                <li className="flex gap-3">
                  <CheckCircle2 className="shrink-0 text-green-600 mt-1" />
                  <div>
                    <h3 className="font-semibold text-gray-900">Ahorro medible</h3>
                    <p className="text-gray-600">
                      Detectamos consumos ocultos/ineficientes y priorizamos acciones de impacto.
                    </p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <BatteryCharging className="shrink-0 text-green-600 mt-1" />
                  <div>
                    <h3 className="font-semibold text-gray-900">Gestión de BESS</h3>
                    <p className="text-gray-600">
                      Ciclos óptimos de carga/descarga y aprovechamiento de tarifas horarias.
                    </p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <LineChartIcon className="shrink-0 text-green-600 mt-1" />
                  <div>
                    <h3 className="font-semibold text-gray-900">Predicción confiable</h3>
                    <p className="text-gray-600">
                      Forecast fotovoltaico y de demanda para operar con menos incertidumbre.
                    </p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <Leaf className="shrink-0 text-green-600 mt-1" />
                  <div>
                    <h3 className="font-semibold text-gray-900">Impacto ESG</h3>
                    <p className="text-gray-600">
                      Menor huella de carbono y soporte a objetivos de sostenibilidad.
                    </p>
                  </div>
                </li>
              </ul>

              <div className="mt-8 grid grid-cols-3 rounded-2xl border bg-gray-50 overflow-hidden">
                <div className="p-4 text-center">
                  <div className="text-2xl font-bold text-gray-900">10–20%</div>
                  <p className="text-xs text-gray-500">Reducción de consumo*</p>
                </div>
                <div className="p-4 text-center border-l">
                  <div className="text-2xl font-bold text-gray-900">24/7</div>
                  <p className="text-xs text-gray-500">Monitoreo</p>
                </div>
                <div className="p-4 text-center border-l">
                  <div className="text-2xl font-bold text-gray-900">CO₂↓</div>
                  <p className="text-xs text-gray-500">Menos emisiones</p>
                </div>
              </div>
              <p className="mt-3 text-[11px] text-gray-400">
                *Rangos orientativos según sector y madurez operativa.
              </p>
            </div>

            <div className="relative">
              <div className="h-full min-h-[320px] rounded-2xl bg-gradient-to-br from-green-600 via-emerald-500 to-teal-500 shadow-lg p-1">
                <div className="h-full w-full rounded-2xl bg-white/10 backdrop-blur grid place-items-center">
                  <div className="text-center text-white px-6">
                    <h3 className="text-2xl font-semibold">Tecnología + Energía</h3>
                    <p className="mt-2 text-white/90">
                      IA aplicada a datos energéticos para decisiones en tiempo real.
                    </p>
                    <a
                      href="#contacto"
                      className="mt-5 inline-flex rounded-2xl bg-white/90 px-4 py-2 text-green-700 font-semibold hover:bg-white"
                    >
                      Conversemos
                    </a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      

      {/* Contacto */}
      <section id="contacto" className="scroll-mt-28 py-20 px-6 bg-gray-50 text-center">
        <h3 className="text-3xl font-bold mb-6">Contáctanos</h3>
        <div className="flex flex-col md:flex-row justify-center gap-8 max-w-4xl mx-auto">
          <div className="space-y-3">
            <p className="flex items-center justify-center gap-2">
              <Linkedin />
              <a className="underline" href="https://www.linkedin.com/company/energia-ia/about/?viewAsMember=true" target="_blank" rel="noreferrer noopener">
                Síguenos en LinkedIn
              </a>
            </p>
            <p className="flex items-center justify-center gap-2">
              <Phone /> <a href="tel:+51949779560">+51 949 779 560</a>
            </p>
            <p className="flex items-center justify-center gap-2">
              <Phone /> <a href="tel:+51947982295">+51 947 982 295</a>
            </p>
            <p className="flex items-center justify-center gap-2">
              <Mail />{" "}
              <a href="mailto:consultas@energiape.com">consultas@energiape.com</a>
            </p>
          </div>
          <form
  className="bg-white shadow rounded-2xl p-6 space-y-4 w-full md:w-1/2"
  onSubmit={onSubmit}
>
  <input
    className="w-full border rounded-2xl px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-300"
    placeholder="Nombre"
    name="nombre"
    value={form.nombre}
    onChange={onChange}
  />
  <input
    className="w-full border rounded-2xl px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-300"
    placeholder="Correo"
    type="email"
    name="correo"
    value={form.correo}
    onChange={onChange}
    required
  />
  <input
    className="w-full border rounded-2xl px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-300"
    placeholder="Empresa"
    name="empresa"
    value={form.empresa}
    onChange={onChange}
  />
  <textarea
    className="w-full border rounded-2xl px-4 py-2 h-28 resize-none focus:outline-none focus:ring-2 focus:ring-green-300"
    placeholder="Mensaje"
    name="mensaje"
    value={form.mensaje}
    onChange={onChange}
    required
  />
  {/* honeypot oculto */}
  <input
    type="text"
    name="hp"
    value={form.hp}
    onChange={onChange}
    style={{ display: "none" }}
    tabIndex={-1}
    autoComplete="off"
  />
  <button
    className="bg-green-600 text-white w-full rounded-2xl py-2 hover:bg-green-700 disabled:opacity-60"
    disabled={sending}
  >
    {sending ? "Enviando..." : "Enviar"}
  </button>
  {okMsg && <p className="text-green-600">{okMsg}</p>}
  {errMsg && <p className="text-red-600">{errMsg}</p>}
</form>

        </div>
      </section>

      {/* Footer */}
      <footer className="bg-green-700 text-white py-6 text-center">
        <p className="font-bold">EnergIA – Inteligencia para un futuro energético sostenible</p>
        <p className="text-sm">© {yearNow} EnergIA. Todos los derechos reservados.</p>
      </footer>
    </div>
  );
}

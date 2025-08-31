import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import logoNegro from '../assets/logoNegro.png';
import { Container, Row, Col } from 'react-bootstrap';

function Home() {
  return (
    <div style={{ marginTop: '80px' }}>
      {/* Sección Soluciones */}
      <section id="soluciones" className="container py-5">
        <h2 className="mb-4 text-center">Soluciones</h2>
        <Row className="g-4">
          <Col md={4}>
            <div className="card h-100 shadow-sm">
              <div className="card-body">
                <h5 className="card-title">Gestión Energética</h5>
                <p className="card-text">
                  Optimizamos el consumo de energía en instalaciones industriales o empresariales,
                  identificando patrones ineficientes y proponiendo mejoras automáticas.
                </p>
              </div>
            </div>
          </Col>
          <Col md={4}>
            <div className="card h-100 shadow-sm">
              <div className="card-body">
                <h5 className="card-title">Predicción de Generación de Energía</h5>
                <p className="card-text">
                  Utilizamos modelos de inteligencia artificial para predecir cuánta energía
                  generarás en base a datos históricos y condiciones operativas.
                </p>
              </div>
            </div>
          </Col>
          <Col md={4}>
            <div className="card h-100 shadow-sm">
              <div className="card-body">
                <h5 className="card-title">Predicción Fotovoltaica</h5>
                <p className="card-text">
                  Calculamos cuánta energía solar podrás producir en función del clima, inclinación,
                  ubicación y datos del sistema fotovoltaico.
                </p>
              </div>
            </div>
          </Col>
        </Row>
      </section>

      {/* Sección Resultados */}
      <section id="resultados" className="bg-light py-5">
        <div className="container">
          <h2>Resultados</h2>
          <p>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis cursus, 
            nunc et facilisis pharetra, lacus libero congue magna, nec laoreet magna eros nec sapien.
          </p>
        </div>
      </section>

      {/* Sección Nosotros */}
      <section id="nosotros" className="container py-5">
        <h2>Nosotros</h2>
        <Row className="align-items-center">
          <Col md={6} className="text-center text-md-start">
            <p>
              Somos una startup que implementa soluciones tecnológicas para reducir
              tus consumos energéticos, tus emisiones de CO2 y ahorrarte dinero.
              Enfocado en metas RSE empresariales.
            </p>
          </Col>
          <Col md={6} className="text-center text-md-middle">
            <img src={logoNegro} alt="Logo EnerglA" height="150" />
          </Col>
        </Row>
      </section>

      {/* Sección Contáctanos */}
      <section id="contacto" className="bg-light py-5">
        <Container>
          <h2 className="mb-4">Contáctanos</h2>
          <Row>
            <Col md={3} className="text-center">
              <i className="bi bi-linkedin fs-1 text-primary"></i>
              <p className="mt-2">
                Síguenos en <a href="#">LinkedIn</a><br />
                para recibir todas las novedades.
              </p>
            </Col>
            <Col md={3} className="text-center">
              <i className="bi bi-telephone-fill fs-1 text-primary"></i>
              <p className="mt-2">
                +51 949 779 560<br />
                +51 947 982 295<br />
              </p>
            </Col>
            <Col md={3} className="text-center">
              <i className="bi bi-envelope-fill fs-1 text-primary"></i>
              <p className="mt-2">
                <a href="mailto:info@analyticsforindustry.com">
                  info@analyticsforindustry.com
                </a>
              </p>
            </Col>
          </Row>
        </Container>
      </section>
    </div>
  );
}

export default Home;

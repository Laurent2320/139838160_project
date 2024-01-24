import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Banner from './Banner';
import Footer from './Footer';
import Navbar from './Navbar';
import Home from './Home';
import PredictGrids from './PredictGrids';
import Statistics from './Statistics';

function App() {
  return (
    <Router>
      <div className="app">
        <Banner />
        <Navbar />
        <div className="content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict-grids" element={<PredictGrids />} />
            <Route path="/stat" element={<Statistics />} />
          </Routes>
        </div>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
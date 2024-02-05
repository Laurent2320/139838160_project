import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

const Home = () => {
  const [maxInvest, setMaxInvest] = useState('');
  const [gridNumber, setGridNumber] = useState('');

  const handleMaxInvestChange = (e) => {
    const value = e.target.value;
    setMaxInvest(value);

    // Add a delay of 400 milliseconds before updating gridNumber
    setTimeout(() => {
      setGridNumber((parseInt(value / 2.5)).toString());
    }, 400);
  };

  const handleGridNumberChange = (e) => {
    const value = e.target.value;
    setGridNumber(value);

    // Add a delay of 500 milliseconds before updating maxInvest
    setTimeout(() => {
      setMaxInvest((parseInt(value) * 2.5).toString());
    }, 500);
  };

  return (
    <div className="home">
      <div className="input-container">
        <label className="input-label">
          Max Invest €:
          <input
            type="text"
            value={maxInvest}
            onChange={handleMaxInvestChange}
            className="input-field"
          />
        </label>
        <label className="input-label">
          Number of Grid:
          <input
            type="text"
            value={gridNumber}
            onChange={handleGridNumberChange}
            className="input-field"
          />
        </label>
        <Link to={`/predict-grids?maxInvest=${maxInvest}&gridNumber=${gridNumber}`}>
          <button className="predict-button">Predict</button>
        </Link>
      </div>
    </div>
  );
};

export default Home;
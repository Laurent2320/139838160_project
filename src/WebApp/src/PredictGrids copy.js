import React from 'react';
import { useLocation } from 'react-router-dom';
import './PredictGrids.css';

const generateGrid = (numberOfGrids) => {
  const grid = [];

  for (let i = 0; i < numberOfGrids; i++) {
    const row = [];
    for (let j = 0; j < 7; j++) {
      const number = i * 7 + j + 1;
      row.push(number);
    }
    grid.push(row);
  }

  return grid;
};

const getRandomNumber = (min, max) => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

const PredictGrids = () => {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const gridNumber = parseInt(queryParams.get('gridNumber')) || 0;
  const grid = generateGrid(gridNumber);

  return (
    <div className="predict-grids">
      <p className='nbr_grid'>{gridNumber} predicted grids</p>
      <div className="grid-container">
        {grid.map((row, rowIndex) => (
          <div
            key={rowIndex}
            className={`row${rowIndex + 1} grid-row`}
          >
            {row.map((number, columnIndex) => (
              <div key={columnIndex} className="grid-cell">
                {number % 7 <= 5 && number % 7 > 0 ? (
                  <>
                    <img src="blue_bubble.png" alt={`Number ${number}`} />
                    <div className="number-overlay">{getRandomNumber(1, 50)}</div>
                  </>
                ) : (
                  <>
                    <img src="yellow_star.png" alt={`Number ${number}`} />
                    <div className="number-overlay">{getRandomNumber(1, 12)}</div>
                  </>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className='space'></div>
    </div>
  );
};

export default PredictGrids;
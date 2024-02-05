import React from 'react';
import { useLocation } from 'react-router-dom';
import './PredictGrids.css';

// const generateGrid = (numberOfGrids) => {
//   const apiUrl = 'http://127.0.0.1:8000/generate_grids';
//   const requestData = {
//     number_of_grid: numberOfGrids
//   };
  
//   return fetch(apiUrl, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify(requestData),
//     })
//       .then(response => {
//         if (!response.ok) {
//           throw new Error('Network response was not ok');
//         }
//         return response.json();
//       })
//       .then(data => {
//         const resultList = data.list_1;
//         const grid = resultList.map(row => [...row]); // Ensure grid is an array
//         console.log('Response:', data);
//         return grid;
//       })
//       .catch(error => {
//         console.error('Error:', error.message);
//         throw error;
//       });
//   };

const generateGrid = async (numberOfGrids) => {
  try {
    const apiUrl = 'http://127.0.0.1:8000/generate_grids';
    const requestData = {
      number_of_grid: numberOfGrids
    };

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    const resultList = data.list_1;
    const grid = resultList.map(row => [...row]); // Ensure grid is an array
    console.log('Response:', data);

    return grid;
  } catch (error) {
    console.error('Error:', error.message);
    throw error;
  }
};

// // Example usage
// generateGrid()
//   .then(myGrid => {
//     console.log('Generated Grid:', myGrid);
//   })
//   .catch(error => {
//     // Handle errors here
//     console.error('Error:', error.message);
//   });


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
import React from 'react';
import { NavLink  } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  return (
    <div className="navbar">
      <nav>
        <ul>
          <li><NavLink exact to="/" activeClassName="active">Home</NavLink></li>
          <li><NavLink to="/predict-grids" activeClassName="active">Predict Grids</NavLink></li>
          <li><NavLink to="/stat" activeClassName="active">Statistics</NavLink></li>
        </ul>
      </nav>
    </div>
  );
};

export default Navbar;
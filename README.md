#solar Power Generation Prediction - Streamlit App

This project aims to predict **solar power generation** using environmental variables such as temperature, wind speed, humidity, and more. The model is built using a **Random Forest Regressor** and deployed using **Streamlit** for interactive predictions.

---

##Objective

To predict solar power generation (in Joules) for every 3 hours based on environmental features.  
This is a **regression** problem where the goal is to model energy production as a function of 9 environmental variables.



## Features Used

- `distance_to_solar_noon` – in radians  
- `temperature` – daily average in °C  
- `wind_direction` – in degrees (0-360)  
- `wind_speed` – daily average in m/s  
- `sky_cover` – scale from 0 (clear) to 4 (covered)  
- `visibility` – in kilometers  
- `humidity` – in percentage  
- `average_wind_speed` – 3-hour average wind speed in m/s  
- `average_pressure` – 3-hour average pressure in inches of mercury  

**Target Variable:**  
- `power_generated` – solar power generated every 3 hours (in Joules)





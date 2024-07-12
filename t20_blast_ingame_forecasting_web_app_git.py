#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:42:06 2024

@author: LeeT19
"""
import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Load the pickled model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def predict_score(over, runs_last_5, team_runs, wickets_last_5, team_wickets, batting_team, bowling_team, venue):
    # Create a dictionary of all the features
    feature_dict = {
        'over': over,
        'runs_last_5': runs_last_5,
        'team_runs': team_runs,
        'wickets_last_5': wickets_last_5,
        'team_wickets': team_wickets,
        f'Batting Team_{batting_team}': 1,
        f'Bowling Team_{bowling_team}': 1,
        f'Venue_{venue}': 1
    }

    # Create a DataFrame with the same structure as the training data
    input_data = pd.DataFrame([feature_dict])

    # Define all columns that should be present in the DataFrame
    all_columns = [
        'over', 'runs_last_5', 'team_runs', 'wickets_last_5', 'team_wickets',
        'Batting Team_Birmingham Bears', 'Batting Team_Derbyshire Falcons', 'Batting Team_Durham Cricket',
        'Batting Team_Essex', 'Batting Team_Glamorgan', 'Batting Team_Gloucestershire', 'Batting Team_Hampshire',
        'Batting Team_Kent Spitfires', 'Batting Team_Lancashire Lightning', 'Batting Team_Leicestershire Foxes',
        'Batting Team_Middlesex', 'Batting Team_Northamptonshire Steelbacks', 'Batting Team_Notts Outlaws',
        'Batting Team_Somerset', 'Batting Team_Surrey', 'Batting Team_Sussex Sharks', 'Batting Team_Worcestershire Rapids',
        'Batting Team_Yorkshire Vikings', 'Bowling Team_Birmingham Bears', 'Bowling Team_Derbyshire Falcons',
        'Bowling Team_Durham Cricket', 'Bowling Team_Essex', 'Bowling Team_Glamorgan', 'Bowling Team_Gloucestershire',
        'Bowling Team_Hampshire', 'Bowling Team_Kent Spitfires', 'Bowling Team_Lancashire Lightning',
        'Bowling Team_Leicestershire Foxes', 'Bowling Team_Middlesex', 'Bowling Team_Northamptonshire Steelbacks',
        'Bowling Team_Notts Outlaws', 'Bowling Team_Somerset', 'Bowling Team_Surrey', 'Bowling Team_Sussex Sharks',
        'Bowling Team_Worcestershire Rapids', 'Bowling Team_Yorkshire Vikings', 'Venue_Cloud County Ground',
        'Venue_County Ground, Northampton', 'Venue_Edgbaston Stadium', 'Venue_Emirates Old Trafford', 'Venue_Headingley',
        'Venue_Incora County Ground', 'Venue_Liverpool CC', "Venue_Lord's Cricket Ground", 'Venue_New Road',
        'Venue_Seat Unique Riverside', 'Venue_Seat Unique Stadium', 'Venue_Sophia Gardens', 'Venue_The 1st Central County Ground',
        'Venue_The Cooper Associates County Ground', 'Venue_The County Ground', 'Venue_The Kia Oval',
        'Venue_The Spitfire Ground, St Lawrence', 'Venue_Trent Bridge', 'Venue_Uptonsteel County Ground', 'Venue_Utilita Bowl'
    ]

    # Fill missing columns with 0
    for col in all_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match the training data
    input_data = input_data[all_columns]

    # Make prediction
    prediction = loaded_model.predict(input_data)
    return prediction[0]

def main():
    st.title('T20 Blast Score Forecasting Tool')
    
    # Input fields for numeric values
    over = st.text_input('Current Over', '0')  # Default to '0' if empty
    runs_last_5 = st.text_input('Runs Scored Last 5 Overs', '0')  # Default to '0' if empty
    team_runs = st.text_input('Current Team Runs', '0')  # Default to '0' if empty
    wickets_last_5 = st.text_input('Wickets Last 5 Overs', '0')  # Default to '0' if empty
    team_wickets = st.text_input('Current Wickets Lost', '0')  # Default to '0' if empty
    
    # Dropdown for Batting Team
    batting_teams = [
        'Birmingham Bears', 'Derbyshire Falcons', 'Durham Cricket', 'Essex', 'Glamorgan', 'Gloucestershire', 'Hampshire',
        'Kent Spitfires', 'Lancashire Lightning', 'Leicestershire Foxes', 'Middlesex', 'Northamptonshire Steelbacks',
        'Notts Outlaws', 'Somerset', 'Surrey', 'Sussex Sharks', 'Worcestershire Rapids', 'Yorkshire Vikings'
    ]
    batting_team = st.selectbox('Batting Team', options=batting_teams)

    # Dropdown for Bowling Team
    bowling_teams = [
        'Birmingham Bears', 'Derbyshire Falcons', 'Durham Cricket', 'Essex', 'Glamorgan', 'Gloucestershire', 'Hampshire',
        'Kent Spitfires', 'Lancashire Lightning', 'Leicestershire Foxes', 'Middlesex', 'Northamptonshire Steelbacks',
        'Notts Outlaws', 'Somerset', 'Surrey', 'Sussex Sharks', 'Worcestershire Rapids', 'Yorkshire Vikings'
    ]
    bowling_team = st.selectbox('Bowling Team', options=bowling_teams)

    # Dropdown for Venue
    venues = [
        'Cloud County Ground', 'County Ground, Northampton', 'Edgbaston Stadium', 'Emirates Old Trafford', 'Headingley',
        'Incora County Ground', 'Liverpool CC', "Lord's Cricket Ground", 'New Road', 'Seat Unique Riverside',
        'Seat Unique Stadium', 'Sophia Gardens', 'The 1st Central County Ground', 'The Cooper Associates County Ground',
        'The County Ground', 'The Kia Oval', 'The Spitfire Ground, St Lawrence', 'Trent Bridge', 'Uptonsteel County Ground',
        'Utilita Bowl'
    ]
    venue = st.selectbox('Venue', options=venues)

    # Code for prediction
    score_prediction = ''
    
    # Creating a button for prediction
    if st.button('Score Prediction'):
        try:
            # Convert inputs to appropriate types before prediction
            over = int(over)
            runs_last_5 = int(runs_last_5)
            team_runs = int(team_runs)
            wickets_last_5 = int(wickets_last_5)
            team_wickets = int(team_wickets)

            # Make prediction
            score_prediction = predict_score(over, runs_last_5, team_runs, wickets_last_5, team_wickets, batting_team, bowling_team, venue)
            
            st.success(f'Predicted Score: {score_prediction}')
        except Exception as e:
            st.error(f'Error: {str(e)}')

if __name__ == '__main__':
    main()











    

'''
12/12/2023
Code to visualize redistricting of Rhode Island based on predictive model with gerrymandering metric inputs
authors: Mac Barnes and Kayleigh Crow

to run: call '!streamlit run 'streamlitModelGA.py' in a notebook

'''
import streamlit as st
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import pandas as pd
import json
import tensorflow as tf
import numpy as np
import branca.colormap as cm
import seaborn as sns
import branca 

primaryColor="#FF8B3D"

# DEFINITIONS

def model_predict():
    model_file = 'GA_UPDATED_Decoder.h5' 
    user_input = [cut_edges_input, pf_input]
    user_input = tf.convert_to_tensor(user_input)
    user_input = user_input * 0.01
    user_input = tf.reshape(user_input, [1, 2])
    noise = tf.random.uniform(shape=[1, 1024])
    decoder = tf.keras.models.load_model(model_file) 
    prediction = decoder.predict([noise, user_input])
    prediction = np.argmax(prediction, axis=-1)
    prediction = prediction + 1
    predictiondf = pd.DataFrame(prediction)
    predictiondf.to_csv("assignments.csv", index=False) #logging for debug
    return predictiondf

def generate_plan(precinct_shp):
    precs = gpd.read_file(precinct_shp)
    precinct_names = precs['PRECINCT_N']
    IDs = precs['ID']
    assignment = model_predict().iloc[0]
    pef = pd.DataFrame({'ID': IDs, 'PRECINCT_N': precinct_names, 'assignment': assignment})
    # Select only necessary columns from precs
    precs_selected = precs[['ID', 'PRECINCT_N', 'geometry']]
    # Merge DataFrames using 'ID' as the common column
    precs_merged = precs_selected.merge(pef, on='ID', how='left', suffixes=('_precs', '_pef'))
    # Group by 'assignment' and dissolve geometry
    precs_dissolve = precs_merged.groupby('assignment').geometry.apply(lambda x: x.unary_union).reset_index()
    # Create GeoDataFrame
    precs_dissolve_gdf = gpd.GeoDataFrame(precs_dissolve, geometry='geometry')
    # Save to CSV
    precs_dissolve_gdf.to_csv("pred_test.csv", index=False) #logging for debug

    return precs_dissolve_gdf


def create_map(precinct_shp):
    # RI lat lon
    lat = 33.247875
    lon = -83.441162
    
    #read in data
    geo_data = generate_plan(precinct_shp)

    # Convert GeoDataFrame to GeoJSON format for compatability 
    geo_json_data = geo_data.to_json()
    geo_json_dict = json.loads(geo_json_data)
    
    # Specify the file path
    json_file_path = "json4map.json"

    # Open the JSON file in write mode
    with open(json_file_path, mode='w') as json_file:
        # Write the geo_json_dict to the JSON file
        json.dump(geo_json_dict, json_file)

    
    # Get a list of 'assignment' values from GeoJSON features
    assignments = [feature['properties']['assignment'] for feature in geo_json_dict['features']]
    asslist = np.array(assignments)
    np.savetxt("asstest.csv", asslist, delimiter = ',') #logging for debug

    # Create a color ramp based on the unique assignments
    unique_assignments = list(set(assignments))
    color_ramp = sns.color_palette("Blues", n_colors=len(unique_assignments))

    # Create a color_dict mapping assignment values to colors
    color_dict = {assignment: f"rgb({int(color_ramp[i][0] * 255)}, {int(color_ramp[i][1] * 255)}, {int(color_ramp[i][2] * 255)})" 
                  for i, assignment in enumerate(unique_assignments)}

    # Specify the file path
    json_file_path = "color_dict.json"

    # Save the dictionary to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(color_dict, json_file)
        
    #initialize folium map
    m = folium.Map(location=[lat, lon], tiles="CartoDB positron", name="Light Map",
                   zoom_start=7, attr="My Data Attribution")

    # Add GeoJSON layer to the map
    folium.GeoJson(
        geo_json_data,
        style_function=lambda feature: {
            "fillColor": color_dict.get(feature['properties']['assignment'], "default_color"),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
            "lineOpacity": 0.1,
        },
        smooth_factor=2.0,
        #highlight_function=lambda x: {'weight': 2, 'color': 'black'},
    ).add_to(m)
    
    color_ramp = branca.colormap.LinearColormap(colors=color_ramp, index=unique_assignments)
    color_ramp.caption = "Distinct Districts"
    color_ramp.add_to(m)
    folium.LayerControl().add_to(m)

    return m

# SIDEBAR PREFERENCES

st.sidebar.title("Choose A State:")
state_mode = st.sidebar.selectbox(
    'What state would you like to generate a map for?',
    ['Georgia']
)

st.sidebar.title("Decide Your Metrics:")
st.sidebar.subheader("Cut Edges")
cut_edges_input = st.sidebar.slider(
    'Select a cut edges score',
    574.0, 621.0
)
st.sidebar.subheader("Partisan Fairness")
pf_input = st.sidebar.number_input(
    "Desired Partisan Fairness score"
)
st.sidebar.title("Generate Map:")
generate = st.sidebar.button("Generate Map")
st.sidebar.header("Download Map")
shape_dl = st.sidebar.checkbox("Download .SHP")
geojson_dl = st.sidebar.checkbox("Download .GEOJSON")
pef_dl = st.sidebar.checkbox("Download Equivalency File")
download = st.sidebar.button("Download Files")
st.markdown("<h1 style='text-align: center; color: black;"
            "'>Set Preferences and Click Generate Map to see Districting Plan 🗺️ </h1>",
            unsafe_allow_html=True)

generate_plan('GA_precincts16.shp')
create_map('GA_precincts16.shp')

folium_static(create_map('GA_precincts16.shp'), width=850, height=500)
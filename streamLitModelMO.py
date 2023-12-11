import streamlit as st
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import pandas as pd
import json
import tensorflow as tf
import numpy as np

primaryColor="#FF8B3D"

# DEFINITIONS
def model_predict():
    model_file = 'mulligans_decoder.h5' #replace with new output 
    user_input = [cut_edges_input, pf_input]
    user_input = tf.convert_to_tensor(user_input)
    user_input = user_input * 0.01
    user_input = tf.reshape(user_input, [1, 2])
    noise = tf.random.uniform(shape=[1, 1024])
    decoder = tf.keras.models.load_model(model_file) 
    prediction = decoder.predict([noise, user_input])
    prediction = np.argmax(prediction, axis=-1)
    prediction = prediction + 1
    prediction = np.array(prediction)
    return prediction

def generate_plan():
    lat = 38.573936
    lon = -92.603760
    precinct_shapefile = 'mo_vtds_geo.geojson'  # MO input data 
    #assignment_file = 'mo_vtds_csv.csv'  
    labels = "MOnames.json"

    #precinct_names = pd.read_json(labels, lines = True)
    IDs = pd.read_csv('MOID.csv')

    # Ensure 'ID' column is numeric
    IDs['ID'] = pd.to_numeric(IDs['ID'], errors='coerce')
    # Drop rows with NaN in 'ID' column
    IDs = IDs.dropna(subset=['ID'])

    IDsarr = np.array(IDs)
    IDsarr = IDsarr[:,0]
    preds = np.ones(len(IDsarr)) #DUMMY PREDICTIONS FOR MO
    #np.savetxt("pred_test.csv", IDsarr, delimiter=",")
    predsIds = np.concatenate((IDsarr.reshape(-1, 1), preds.reshape(-1, 1)), axis=1)

    #df to CSV
    np.savetxt("pred_test.csv", predsIds, delimiter=",")
    preds_df = pd.DataFrame(predsIds)
    preds_df.columns = ['ID', 'assignment']
    preds_df.to_csv('predsdf.csv')
    #prediction_df = pd.DataFrame(prediction_array, columns=['assignment'])
   # Combine GeoDataFrame with DataFrame
    nc_precs = gpd.read_file(precinct_shapefile)
    #nc_precs_join = nc_precs.merge(preds_df, left_on='loc_prec', right_on='ID', how='left')
    nc_precs_join = pd.concat([nc_precs, preds_df])
  
    nc_precs_dissolve = nc_precs_join.dissolve(by='assignment')
    nc_precs_dissolve['assign'] = nc_precs_dissolve.index.copy()
    null_columns = nc_precs_dissolve.columns[nc_precs_dissolve.isnull().any()]
    return nc_precs_dissolve

def create_map():

    lat = 38.573936
    lon = -92.603760
    precinct_shapefile = 'mo_vtds_geo.geojson'  # Update with the actual file name
    assignment_file = 'mo_vtds_csv.csv'  # Update with the actual file name
    labels = "MOnames.json"

    m = folium.Map(location=[lat, lon], tiles="CartoDB positron", name="Light Map",
                   zoom_start=7, attr="My Data Attribution")
    folium.Choropleth(
        geo_data=generate_plan(),
        name="choropleth",
        data=generate_plan(),
        columns=["assign", "ID"],
        key_on='feature.properties.assign',
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.1,
    ).add_to(m)
    return m


# SIDEBAR PREFERENCES

st.sidebar.title("Choose A State:")
state_mode = st.sidebar.selectbox(
    'What state would you like to generate a map for?',
    ['Missouri']
)

st.sidebar.title("Decide Your Metrics:")
st.sidebar.subheader("Cut Edges")
cut_edges_input = st.sidebar.slider(
    'Select a cut edges score',
    574.0, 621.0
)
st.sidebar.subheader("Mean-Median")
pf_input = st.sidebar.number_input(
    "Desired mean-median score"
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

generate_plan()
create_map()

folium_static(create_map(), width=850, height=500)

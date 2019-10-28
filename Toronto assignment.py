#!/usr/bin/env python
# coding: utf-8

# # Part 1. Assignment week 3 - Cleaning dataset

# In[12]:


get_ipython().system('conda install urllib2')


# In[24]:


#Download packages
import pandas as pd
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup


# Scrape the webpage

# In[136]:


url="https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
req = urlopen(url)
soup = BeautifulSoup(req,"lxml")
right_table=soup.find('table', class_='wikitable sortable')


# Create a dataframe

# In[115]:


A=[]
B=[]
C=[]
for row in right_table.findAll('tr'):
    cells=row.findAll('td')
    if len(cells)==3:
        A.append(cells[0].find(text=True))
        B.append(cells[1].find(text=True))
        C.append(cells[2].find(text=True))


# In[114]:


import pandas as pd
df=pd.DataFrame(A,columns=['Postcode'])
df['Borough']=B
df['Neighbourhood']=C
df = df[df.Borough!="Not assigned"]


# Group by Borough

# In[130]:


df = df.groupby(['Postcode', 'Borough'])['Neighbourhood'].apply( ', '.join).str.replace('(\\n)', '').reset_index()


# Assign the Borough names to Neighbourhoods where names are not specified

# In[134]:


import numpy as np
df['Neighbourhood'] = np.where(df['Neighbourhood'] == 'Not assigned', df['Borough'], df['Neighbourhood'])


# In[156]:


df.shape


# # Part 2. Assignment week 3 - Getting location data

# In[146]:


#read the file
geodata = pd.read_csv("http://cocl.us/Geospatial_data")
geodata


# In[163]:


df1 = pd.merge(df, geodata, on=geodata['Postal Code'])
df1 = df1.drop('key_0', 1)
df1 = df1.drop('Postal Code', 1)
df1


# # Part 3. Assignment week 3 - Clustering

# In[167]:


from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library


# Plot Toronto map

# In[169]:


address = 'Toronto, ON'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# In[173]:


map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighbourhood in zip(df1['Latitude'], df1['Longitude'], df1['Borough'], df1['Neighbourhood']):
    label = '{}, {}'.format(neighbourhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# Define Foursquare Credentials and Version

# In[174]:


CLIENT_ID = 'JAM12CPOMRFSVKPNQOEXBKBJEUENNHOZQW2GYLDYHJR1FGIQ' # your Foursquare ID
CLIENT_SECRET = 'WBBFTIOJF2NJE3T4TZVFGBWOQGE4TLEEJ4MIKE14OBTHIAOV' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# Get the venues in the radius of 500

# In[175]:


LIMIT=100
radius=500
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    latitude, 
    longitude, 
    radius, 
    LIMIT)
url


# In[177]:


results = requests.get(url).json()


# In[189]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[190]:


venues = getNearbyVenues(names=df1['Neighbourhood'],
                                   latitudes=df1['Latitude'],
                                   longitudes=df1['Longitude']
                                  )


# In[191]:


venues.head()


# In[192]:


print('{} venues were returned by Foursquare.'.format(venues.shape[0]))


# Process the venues data

# In[193]:


venues.groupby('Neighborhood').count()


# In[196]:


onehot = pd.get_dummies(venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
onehot['Neighborhood'] = venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [onehot.columns[-1]] + list(onehot.columns[:-1])
onehot = onehot[fixed_columns]

grouped = onehot.groupby('Neighborhood').mean().reset_index()
grouped


# Print each Neighbourhood with top 5 locations

# In[197]:


num_top_venues = 5

for hood in grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = grouped[grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# Create a pandas dataframe

# In[198]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# Create a new dataframe with top 10 venues

# In[202]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = grouped['Neighborhood']

for ind in np.arange(grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[203]:


kclusters = 5

grouped_clustering = grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)


# In[224]:


merged = df1

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
merged = merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighbourhood')

#some data was not clustered, therefore, need to drop NaN values
merged.dropna(axis=0, inplace=True)

merged.head(50) # check the last columns!


# In[225]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(merged['Latitude'], merged['Longitude'], merged['Neighbourhood'], merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color = rainbow[int(cluster)-1],
        fill=True,
        fill_color=rainbow[int(cluster)-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:





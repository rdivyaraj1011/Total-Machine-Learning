import streamlit as st

# Set the title of the web app
st.title('Hello Streamlit!')

# Add a header
st.header('This is a header')

# Add a text input
name = st.text_input('Enter your name:')
number = st.number_input('Pick a number:', min_value=0, max_value=100)
slider_value = st.slider('Select a value:', min_value=0, max_value=200, value=50)
option = st.selectbox('Choose an option:', ['Yes', 'No', 'None of these'])
checkbox = st.checkbox('Check me')

# Add a button
if st.button('Button'):
    st.write(f'Hello, {checkbox}!')


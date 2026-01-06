import streamlit as st
import makeData

# Assign the plot object from your module
plt_obj = makeData.plt

st.title("My BMI Data Visualization")

# Instead of plt.show(), use st.pyplot()
# We pass the 'gcf' (Get Current Figure) to streamlit
st.pyplot(plt_obj.gcf())
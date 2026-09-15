import streamlit as st 
import pandas as pd
import numpy as np 
import joblib 
import gzip

with gzip.open("tweetboost_model.joblib.gz","rb") as f:
    model=joblib(f)
model_columns=joblib.load("model_columns.joblib")

st.title("🐦 TweetBoost")
st.write("Predict tweet virality before you post.")
with st.form("tweet_form"):
    st.subheader("Timing Section")
    hour = st.slider("Hour of the Day", 0, 23, 12)
    month = st.selectbox("Month", range(1, 13))
    day = st.number_input("Day of month", 1, 31, 15)
    year = st.selectbox("Year", range(2015, 2025), index=5)
    
    st.subheader("Content Structure")
    hashtag_count = st.slider("Hashtag count", 0, 15, 2)
    url_count = st.number_input("URL count", 0, 5, 0)
    mention_count = st.number_input("Mention count", 0, 10, 0)
    attachment = st.radio("Attachment Type", ["None", "Image", "Video"])
    
    st.subheader("User Features")
    user_id = st.number_input("User ID", 0, 100, 42)
    language_id = st.number_input("Language ID", 0, 10, 0)
    topic_count = st.slider("Topic count", 2, 15, 9)
    
    submitted = st.form_submit_button("Predict Virality")
        
    
    if submitted:
    
        is_late_night = 1 if 0 <= hour <= 5 else 0
        is_evening = 1 if 20 <= hour <= 23 else 0
        has_url = 1 if url_count > 0 else 0
        has_attachment = 1 if attachment in ["Image", "Video"] else 0
        has_image = 1 if attachment == "Image" else 0
        has_video = 1 if attachment == "Video" else 0
        has_no_media = 1 if attachment == "None" else 0
        engagement_score = hashtag_count + url_count + mention_count
    
    
        if hashtag_count == 0:
            hashtag_intensity = 4
        elif 1 <= hashtag_count <= 2:
            hashtag_intensity = 2
        elif 3 <= hashtag_count <= 5:
            hashtag_intensity = 3
        elif 6 <= hashtag_count <= 10:
            hashtag_intensity = 1
        else:
            hashtag_intensity = 0
    
        input_data = pd.DataFrame({
        'tweet_user_id': [user_id],
        'tweet_created_at_year': [year],
        'tweet_created_at_month': [month],
        'tweet_created_at_day': [day],
        'tweet_created_at_hour': [hour],
        'tweet_hashtag_count': [hashtag_count],
        'tweet_url_count': [url_count],
        'tweet_mention_count': [mention_count],
        'tweet_has_attachment': [has_attachment],
        'tweet_language_id': [language_id],
        'topic_count': [topic_count],

        'is_late_night': [is_late_night],
        'is_evening': [is_evening],
        'has_url': [has_url],
        'has_image': [has_image],
        'has_video': [has_video],
        'has_no_media': [has_no_media],
        'engagement_score': [engagement_score],
        'hashtag_intensity': [hashtag_intensity],
    })
    
    
        input_data = input_data[model_columns]
    
        
        prediction=model.predict(input_data)[0]
        probablity=model.predict_proba(input_data)[0]
        st.subheader("Prediction Result")
        if prediction==1:
            print("This tweet is likely to go VIRAL!")
        else:
            print("This tweet is unlikely to go viral!")
        viral_prob = probablity[1] * 100
        not_viral_prob = probablity[0] * 100

        st.metric("Viral Probability", f"{viral_prob:.1f}%")
        st.metric("Not Viral Probability", f"{not_viral_prob:.1f}%")

        st.progress(viral_prob / 100)
        st.subheader("Advice")
        if attachment == "None":
            st.write("Text-only tweets perform best in this dataset.")
        elif attachment == "Video":
            st.write("Video tweets have the lowest virality rate. Consider text-only.")
        else:
            st.write("Image tweets perform moderately well.")

        if hashtag_count > 10:
            st.write("Too many hashtags may hurt virality.")
        elif hashtag_count == 0:
            st.write("Adding 1-2 hashtags may help.")
        else:
            st.write("Your hashtag count is in a good range.")
        st.caption("Note: This is a predictive model trained on historical data. Predictions are probabilistic, not guarantees.")
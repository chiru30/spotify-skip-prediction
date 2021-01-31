import streamlit as st
import pickle

lbm_model = pickle.load(open('llbm.pkl','rb'))

def main():
    import streamlit as st
    html_temp = """
    <div style="background-color:#DCE1E3 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Spotify Skip Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("""
This app predicts the **whether the song is skipped or not** !
""")

 
    session_position = st.slider('session_position', 0,20)
    hubrer = st.selectbox('hist_user_behavior_reason_end_remote', (0.0, 1.0))
    valence= st.slider('valence', 0.0, 1.0)
    a0 = st.slider('acoustic_vector_0', -1.0, 1.0)
    a1 = st.slider('acoustic_vector_1', -1.0, 1.0)
    a2 = st.slider('acoustic_vector_2', -1.0, 1.0)
    a3 = st.slider('acoustic_vector_3', -1.0, 1.0)
    a4 = st.slider('acoustic_vector_4', -1.0, 1.0)
    a5 = st.slider('acoustic_vector_5', -1.0, 1.0)
    a6 = st.slider('acoustic_vector_6', -1.0, 1.0)
    a7 = st.slider('acoustic_vector_7', -1.0, 1.0)
    context_type_charts= st.selectbox('context_type_charts', (0.0, 1.0))
    ctep = st.selectbox('context_type_editorial_playlist', (0.0, 1.0))
    ctpp = st.selectbox('context_type_personalized_playlist', (0.0, 1.0))
    ctr = st.selectbox('context_type_radio', (0.0, 1.0))
    ctuc = st.selectbox('context_type_user_collection',(0.0, 1.0))
    hubrsb = st.selectbox('hist_user_behavior_reason_start_backbtn', (0.0, 1.0))
    hubrsc = st.selectbox(' hist_user_behavior_reason_start_clickrow', (0.0, 1.0))
    hubrsf = st.selectbox('hist_user_behavior_reason_start_fwdbtn', (0.0, 1.0))
    
    hubrsr = st.selectbox('hist_user_behavior_reason_start_remote', (0.0, 1.0))
    hubrtrackdone = st.selectbox('hist_user_behavior_reason_start_trackdone', (0.0, 1.0))
    hubree  = st.selectbox('hist_user_behavior_reason_end_endplay', (0.0, 1.0))
    hubref = st.selectbox('hist_user_behavior_reason_end_fwdbtn', (0.0, 1.0))
    hubrel = st.selectbox('hist_user_behavior_reason_end_logout', (0.0, 1.0))
    speechiness = st.slider('speechiness', 0.0, 1.0)
    organism = st.slider('organism', 0.0, 1.0)
    hubret = st.selectbox('hist_user_behavior_reason_end_trackdone', (0.0, 1.0))
    mechanism = st.slider('mechanism', 0.0, 1.0)
    session_length = st.slider('session_length', 10, 20)
    context_switch = st.selectbox('context_switch', (0.0, 1.0))
    npbp = st.selectbox('no_pause_before_play', (0.0, 1.0))
    spbp = st.selectbox('short_pause_before_play',(0.0, 1.0))
    seek_f = st.slider('hist_user_behavior_n_seekfwd', 0.0, 60.0)
    seek_b = st.slider('hist_user_behavior_n_seekback', 0.0, 150.0)
    shuffle = st.selectbox('hist_user_behavior_is_shuffle', (0.0, 1.0))
    hr = st.slider('hour_of_day', 0.0, 24.0)
    mode = st.selectbox('mode', (0.0, 1.0))
    duration = st.slider('duration', 30.0, 1800.0)
    
    premium = st.selectbox('premium', (0.0, 1.0))
    est = st.slider('us_popularity_estimate', 90.0, 100.0)
    loud = st.slider('loudness', -25.0, 0.0)
    live = st.slider('liveness', 0.0, 1.0)
    instrument = st.slider('instrumentalness', 0.0, 1.0)
    flat = st.slider('flatness', 0.5, 1.0)
    release_year = st.slider("release_year:", 1950, 2019)
    energy = st.slider('energy', 0.0, 1.0)
    dance = st.slider('danceability', 0.0, 1.0)
    bounce = st.slider('bounciness', 0.0, 1.0)
    beat_str = st.slider('beat_strength', 0.0, 1.0)
    acous = st.slider('  acousticness', 0.0, 1.0)
    rangemean = st.slider('dyn_range_mean', 0.0, 20.0)
    tempo = st.slider(' tempo', 50.0, 220.0)
    st_end = st.selectbox(' hist_user_behavior_reason_start_endplay', (0.0, 1.0))
    st_play = st.selectbox(' hist_user_behavior_reason_start_playbtn', (0.0, 1.0))
    
    inputs=[[session_position,
             hubrer,
             valence,
             a0,a1,a2,a3,a4,a5,a6,a7,
             context_type_charts,
             ctep,
             ctpp,
             ctr,ctuc,hubrsf,
             hubrsr,
             hubrtrackdone,
             hubree,
             hubref,
             hubrel,
             speechiness,
             organism,
             hubret,
             mechanism,
             session_length,
             context_switch,
             npbp,
             spbp,
             seek_f,
             seek_b,
             shuffle,
             hr,
             mode,
             duration,
             premium,
             est,
             loud,
             live,
             instrument,
             flat,
             release_year,
             energy,
             dance,
             bounce,
             beat_str,
             acous,
             rangemean,
		tempo,
             st_end,st_play
             ]]

    if st.button('Predict'):

         st.success(Predict(lbm_model.predict(inputs)))

def Predict(num):
    if num == 1:
        return 'Skipped'
    else:
        return 'Not-Skipped'
if __name__=='__main__':
    main()

import cv2
import sys
from sklearn.preprocessing import MinMaxScaler
from spellchecker import SpellChecker
from pathlib import Path
import mediapipe as mp
import joblib
import streamlit as st
import pickle
spell = SpellChecker()
mmodel = joblib.load('model.joblib') 

drawingmp_pts = mp.solutions.drawing_utils
drawing_stylesmp_pts = mp.solutions.drawing_styles
handsmp_pts = mp.solutions.hands

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

def load_styles():
    with open("styles/styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
try:
    sys.path.remove(str(parent))
except ValueError:  
    pass

load_styles()

def draw_main_page():
    st.write(
        f"""
        # Welcome to Signcom 👋
        Get started by configuring settings from below.
        """
    )
    facing_cam = 0
    if st.checkbox('Switch camera'):
        facing_cam = 1 - facing_cam
    
    st.subheader('Toggle `Start` to start camera and begin interpreting gestures.') 
    st.write('`del` is used to delete the last symbol entered, `space` is used to end a word.')
    rem_slider_opt = {
        "Stop":"",
        "Start":""
    }
    
    run = (st.select_slider('', ["Stop","Start"],key="slider1", format_func=rem_slider_opt.get))
    st.write("Start")
    if run == "Start":
        hpoints = st.checkbox('Draw landmarks')

    CAM_FRAME = st.image([]) 
    capturr = cv2.VideoCapture(facing_cam)

    wordh = ""
    lastl = ""
    conl = 0
    show_on_screen = False
    new_wrd = False
    correct = 0

    while run == "Start":
        with handsmp_pts.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            while capturr.isOpened():
                ok_cpt , image = capturr.read()
                if not ok_cpt:
                    continue
                width, height = int(capturr.get(3)), int(capturr.get(4))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image.flags.writeable = False
                font = cv2.FONT_HERSHEY_DUPLEX  
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    hand_landmarks, handedness = results.multi_hand_landmarks[0], results.multi_handedness[0]

                    if(hpoints):
                         drawingmp_pts.draw_landmarks(
                           image,
                           hand_landmarks,
                           handsmp_pts.HAND_CONNECTIONS,
                           drawing_stylesmp_pts.get_default_hand_landmarks_style(),
                           drawing_stylesmp_pts.get_default_hand_connections_style())

                    # add text
                    xis = int(hand_landmarks.landmark[0].x * width) - 150 
                    yis = int(hand_landmarks.landmark[0].y * height) - 100 

                    if handedness.classification[0].label == 'Right':
                        xis += 250
                    scaler = MinMaxScaler()

                    x_ogn = hand_landmarks.landmark[0].x
                    y_ogn = hand_landmarks.landmark[0].y
                    z_ogn = hand_landmarks.landmark[0].z

                    pnts_to_show = []
                    for pnt in hand_landmarks.landmark:
                        pnt.x -= x_ogn
                        pnt.y -= y_ogn
                        pnt.z -= z_ogn
                        pnts_to_show.append([pnt.x, pnt.y, pnt.z])
                    
                    scaler_val = scaler.fit_transform(pnts_to_show)

                    res_pnt = []
                    for coord in scaler_val:
                        res_pnt.append(coord[0])
                        res_pnt.append(coord[1])
                        res_pnt.append(coord[2])
                    
                    symbol_mean = mmodel.predict([res_pnt])[0]

                    if symbol_mean == 'del':
                        lastl = ''
                    elif symbol_mean == lastl:
                        conl += 1
                        if conl > 5 and not show_on_screen:
                            if symbol_mean == 'space' and not new_wrd:
                                new_wrd = True
                                mspell = list(spell.unknown([wordh]))

                                if len(mspell) != 0:
                                    wordh = spell.correction(mspell[0])
                                    correct = 1
                                else:
                                    correct = 2

                            elif new_wrd:
                                new_wrd = False
                                correct = 0
                                wordh = symbol_mean
                            else:
                                wordh += symbol_mean

                            show_on_screen = not show_on_screen
                    elif symbol_mean != lastl:
                        conl = 0
                        show_on_screen = False

                    lastl = symbol_mean

            
                    display_letter = symbol_mean if symbol_mean != 'del' else 'next'
                    image = cv2.putText(image, symbol_mean, 
                                (xis, yis),
                                font, 3, (0, 0, 0), 5, cv2.LINE_AA 
                    )

                text_size, _ = cv2.getTextSize(wordh, font, 3, 2) 
                image = cv2.rectangle( 
                        image,
                        (int((width - text_size[0]) / 2), height - 100 - text_size[1] - 30),
                        (int((width + text_size[0]) / 2), height - 130 + text_size[1]),
                        (219, 216, 248), -1)

                display_word = wordh

                if correct == 1:
                    cv2.putText( 
                            image,
                            display_word,
                            (int((width - text_size[0]) / 2), height - 100),
                            font, 3, (0, 255, 255), 2, cv2.LINE_AA) 
                elif correct == 2: 
                    cv2.putText( 
                            image,
                            display_word,
                            (int((width - text_size[0]) / 2), height - 100),
                            font, 3, (0, 255, 0), 2, cv2.LINE_AA) 
                else:
                    cv2.putText( 
                            image,
                            display_word,
                            (int((width - text_size[0]) / 2), height - 100),
                            font, 3, (255, 255, 255), 2, cv2.LINE_AA) 

                CAM_FRAME.image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
    
    else:
        capturr.release()
        cv2.destroyAllWindows()



manual_pg = ""

st.sidebar.header('Signcom', anchor="signcom_title")
st.sidebar.image("assets/menu.png", use_column_width=True)
st.sidebar.write('A tool that uses AI to help convert sign language to speech in realtime.', anchor="signcom_about")
draw_main_page()
    

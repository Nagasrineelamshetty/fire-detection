# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import time

# st.set_page_config(page_title="Fire Detection", layout="centered")

# st.title(" Real-Time Fire Detection ")
# st.markdown("Using a YOLO model to detect fire through the webcam.")

# if 'model' not in st.session_state:
#     st.session_state.model = YOLO('best.pt')

# start_button = st.button("Start Detection")

# if start_button:
#     stframe = st.empty()
#     cap = cv2.VideoCapture(0)

#     stop_button = st.button("Stop Detection", key="stop_button")  # Unique key for stop button
#     fire_detected = False  # To track fire detection status
    
#     # Create a progress bar
#     progress = st.progress(0)  # Initial progress bar value set to 0 (No fire)

#     fire_message = st.empty()  # To display the fire detection status message

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to access webcam.")
#             break

#         results = st.session_state.model.predict(source=frame, imgsz=640, conf=0.6, verbose=False)
#         annotated_frame = results[0].plot()

#         # Check if fire is detected
#         fire_detected = False
#         for result in results[0].boxes.cls:
#             if result == 0:  # Assuming class 0 corresponds to 'fire' in your model
#                 fire_detected = True
#                 break

#         # Update progress bar and message based on detection status
#         if fire_detected:
#             progress.progress(100)  # Full progress bar indicates fire detected
#             fire_message.write("🔥 Fire Detected!")
#         else:
#             progress.progress(0)  # No progress bar means no fire detected
#             fire_message.write("No Fire Detected")

#         # Display the annotated frame with detection
#         stframe.image(annotated_frame, channels="BGR")

#         # Check if the stop button is pressed and break the loop
#         if stop_button:
#             cap.release()
#             st.success("Detection stopped.")
#             break

#         time.sleep(0.03)  # Add a small delay to manage resource usage

import streamlit as st
from ultralytics import YOLO
import cv2
import time
from threading import Thread

st.set_page_config(page_title="Fire Detection", layout="centered")

st.title("Real-Time Fire Detection")
st.markdown("Using a YOLO model to detect fire through the webcam.")

# Threaded Video Capture class to improve FPS
class VideoCaptureAsync:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        Thread(target=self.update, args=()).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# Load model once in session state
if 'model' not in st.session_state:
    st.session_state.model = YOLO('best.pt')

start_button = st.button("Start Detection")
stop_button = st.button("Stop Detection", key="stop_button")

if start_button:
    cap_async = VideoCaptureAsync(0)
    stframe = st.empty()
    progress = st.progress(0)
    fire_message = st.empty()
    fps_display = st.empty()

    fire_detected = False
    frame_counter = 0
    last_time = time.time()
    fps = 0
    annotated_frame = None  # Store last annotated frame

    while True:
        ret, frame = cap_async.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        frame_counter += 1

        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (320, 240))

        if frame_counter % 3 == 0:
            # Run detection every 3 frames
            results = st.session_state.model.predict(source=frame_resized, imgsz=320, conf=0.6, verbose=False)
            annotated_frame = results[0].plot()

            fire_detected = any(cls == 0 for cls in results[0].boxes.cls)  # Assuming class 0 = fire

            # Update progress bar and message
            if fire_detected:
                progress.progress(100)
                fire_message.markdown("🔥 **Fire Detected!**")
            else:
                progress.progress(0)
                fire_message.markdown("No Fire Detected")

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time)
            last_time = current_time

        # Display the last annotated frame scaled back for UI
        if annotated_frame is not None:
            display_frame = cv2.resize(annotated_frame, (640, 480))
            stframe.image(display_frame, channels="BGR")

        fps_display.markdown(f"**FPS:** {fps:.2f}")

        # Stop button check
        if stop_button:
            cap_async.stop()
            st.success("Detection stopped.")
            break

        # Small sleep to reduce CPU usage
        time.sleep(0.01)

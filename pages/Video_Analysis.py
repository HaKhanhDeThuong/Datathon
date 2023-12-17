import streamlit as st
import sys
sys.path.append(r'D:\code_folder\data-code\Datathon2023\git\DatathonFinal\models')
from detector.Yolo_Detector import ObjectDetector
from detector.FasterRNN_Detector import ObjectDetector as OBF
import cv2

st.set_page_config(page_title="Video Analysis", page_icon="images/logo.png",layout="wide")
def title(url):
    st.markdown(f'<p style="font-size:50px; padding: 5px; font-weight: bold; text-align: center; color:#289df2">{url}</p>', unsafe_allow_html=True)

def subheader(url):
    st.markdown(f'<p style="font-size:30px; padding: 5px; font-weight: bold; text-align: center; color:#4F46E5">{url}</p>', unsafe_allow_html=True)
def normalText(url):
    st.markdown(f'<p style="font-size:20px; padding: 5px;  text-align: center; color:#ffffff">{url}</p>', unsafe_allow_html=True)
class VideoAnalysis:
      def __init__(self):
        self.object_detector = ObjectDetector() 
        self.camera_detector = OBF()
      def main(self):
         title("Object Detection and Tracking")
         normalText("Video tracking and object detection of people in front of window displays aim to optimize sales by analyzing customer behavior and preferences for targeted and effective marketing strategies.")
         with st.container(): 
            st.write("---")
         # Choose video source: File upload, video file, or camera
         video_source = st.radio("Select video source", ["File Upload", "Video File", "Camera"])

         if video_source == "File Upload":
               uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mpg"])
               if uploaded_file is not None:
                  video_path = "uploaded_video.mpg"
                  subheader("Video result after detected")

                  with open(video_path, "wb") as f:
                     f.write(uploaded_file.read())
                     cap = cv2.VideoCapture(video_path)
                     if not cap.isOpened():
                           st.error("Error opening video file.")
                           return

                     stframe = st.empty() 
                     while True:
                           ret, frame = cap.read()
                           frame = self.object_detector.image_with_bbox(frame, self.object_detector.detect(frame))
                           if not ret:
                              break
                           stframe.image(frame, channels="BGR", use_column_width=True)


         elif video_source == "Video File":
               video_path = st.text_input("Enter video file path:")
               if video_path:
                  self.object_detector.mp4_loader(video_path)

         elif video_source == "Camera":
               self.camera_detector.video_capture()

if __name__ == "__main__":
   app = VideoAnalysis()
   app.main()
   
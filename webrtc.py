import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import HandTracker as htm

detector = htm.HandDetector()

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = detector.drawHands(img=img)

        return img


webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
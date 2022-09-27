# njn_doorcam

Demo Application for Nvidia Jetson Nano. Face recognition and door greeting robot including voice activation for registering personalised greetings.

Original version at https://git.its.aau.dk/WW82ZE/njn_doorcam

What can you do with Jetson Nano?

There are many examples of projects on Jetson Nano among the Jetson Community Projects https://developer.nvidia.com/embedded/community/jetson-projects

After looking at these projects, you may have some ideas of what is possible. You might even have tried some of the projects on my own Jetson Nano. But how exactly can you start working on my own project? 

In order to help overcome the hurdle of getting started, this writeup goes through the process of creating a project in Python. 

# Making a simple AI application with the camera

When I get home after a long day, I would like to get a smile on my face. One way to do that is if someone will greet me when I get back home. However, if there is nobody home that person could be Jetson Nano. Therefore, we will try to use face recognition using the camera to greet people at the door.

The project will consist of two phases. First we would like to get face recognition working, and then we want to make the system self-contained by adding voice commands.

## Outline of the project:

Phase 1: Detecting a person
- Get a picture from the camera
- Apply face recognition to the obtained image
- Play a greeting on the speakers when a person is detected

Phase 2: Personalising the greeting
- Use voice activation for voice command
- Record voice clip
- Assign voice clip to recognized face


## Detecting a person

Every good idea is based on the foundation of other ideas. This project is no exception. We start by looking at a similar project, where Jetson Nano was used to recognize people from a doorbell camera. https://medium.com/@ageitgey/build-a-face-recognition-system-for-60-with-the-new-nvidia-jetson-nano-2gb-and-python-46edbddd7264

This project uses the same basic building blocks as the doorbell camera project, namely, the image acquisition and face detection parts.

First, we want to get an image from the camera into Python so that we can work with it. We do this using OpenCV. 

In OpenCV, this is relatively straightforward. We first create a `VideoCapture` object and then use the `.read()` method to obtain an image frame. The `VideoCapture` object stores video frames from the camera (In this case camera 0), and the method `.read()` returns a boolean `ret` whether a frame was obtained from the queue, and if there was, a `frame`. This frame is in BGR (Blue, Green, Red) format, which is standard for OpenCV.

```python
import cv2

video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()

if ret:
    cv2.imshow('Video', frame)
```

We then want to send the frame into `face_recognition` to, exactly as the name implies, perform face recognition. The size of the image obtained from the camera (1280x960) is too large to get good performance with `face_recognition` on the Jetson Nano. Therefore, we scale the image down prior to recognition. If we change the frame from BGR to RGB format, we can simply use the `face_recognition` on the resulting frame.

```python
import cv2
import face_recognition

video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()
scale = 4

if ret:
    small_frame = cv2.resize(frame, (0,0), fx=1/scale, fy=1/scale)

    # Convert image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the face locations and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    if face_encodings != []:
        print("Person Detected")

    cv2.imshow('Video', frame)

```

From `face_recognition` we can thus gain the location and encoding of the face which is recognized. The encoding is a mapping of the information which the face contains into a space where two encodings of the face of one person should be closer to each other in the encoding space, while the encodings of the faces of different people should have a greater distance from each other. The model is trained so that pictures of the same person should have a distance less than 0.6 to each other.

Now that we have the face locations and encodings, we can display a bounding box around the face so that we can see visually that the face detection is working.

```python
# Draw a box around each face
for face_location, face_encoding in zip(face_locations, face_encodings):
    top, right, bottom, left = face_location
    
    # Scale back up face locations since detection was performed on downscaled image
    top *= scale
    right *= scale
    bottom *= scale
    left *= scale

    # Draw a box around the face, in the color red (cv2 uses BGR) with a thickness of 2 pixels
    cv2.rectangle(frame, (left, top), (right,bottom), (0,0,255), 2)

cv2.imshow('Video', frame)
```

By collecting the image capture and the face recognition into a function, we can run it in a loop. 

```python
def recognize(video_capture, scale):
    ret, frame = video_capture.read()
    if ret:
        small_frame = cv2.resize(frame, (0, 0), fx=1/scale, fy=1/scale)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    

        if face_encodings != []:
            print("Person Detected")

        # Loop through each detected face
        for face_location, face_encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = face_location

            # Scale back up face locations since detection was performed on downscaled image
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale

            # Draw a box around the face, in the color red (cv2 uses BGR) with a thickness of 2 pixels
            cv2.rectangle(frame, (left, top), (right,bottom), (0,0,255), 2)
        
        cv2.imshow('Video', frame)

import cv2
import face_recognition

video_capture = cv2.VideoCapture(0)
scale = 4

while True:
    recognize(video_capture, scale)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

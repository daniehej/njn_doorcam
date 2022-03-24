"""
Doorcam project
"""
import struct
from threading import Thread
from datetime import datetime, timedelta

import numpy as np
import face_recognition
import cv2
import pvporcupine
import pyaudio
import speech_recognition as sr
import simpleaudio as sa
from scipy.io import wavfile

from util import load_known, save_known


def lookup_known_face(known_faces, face_encoding, return_index=False):
    """
    See if this is a face we already have in our face list
    """
    metadata = None

    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if len(known_faces) == 0:
        return None, metadata

    # Calculate the face distance between the unknown face and every face on in our known face list
    # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
    # the more similar that face was to the unknown face.
    
    known_face_encodings = [f['encoding'] for f in known_faces]
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
    best_match_index = np.argmin(face_distances)

    # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
    # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
    # of the same person always were less than 0.6 away from each other.
    # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
    # people will come up to the door at the same time.
    if face_distances[best_match_index] < 0.6:
        # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
        metadata = known_faces[best_match_index]

        # Update the metadata for the face so we can keep track of how recently we have seen this face.
        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1

        # We'll also keep a total "seen count" that tracks how many times this person has come to the door.
        # But we can say that if we have seen this person within the last 5 minutes, it is still the same
        # visit, not a new visit. But if they go away for awhile and come back, that is a new visit.
        if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=5):
            metadata["first_seen_this_interaction"] = datetime.now()
            metadata["seen_count"] += 1

        return best_match_index, metadata
    
    return None, metadata


def register_new_face(known_faces, known_persons, face_encoding, face_image):
    """
    Add a new person to our list of known faces
    """
    # Add the face encoding to the list of known faces
    # Add a matching dictionary entry to our metadata list.
    # We can use this to keep track of how many times a person has visited, when we last saw them, etc.
    known_faces.append({
        "first_seen": datetime.now(),
        "first_seen_this_interaction": datetime.now(),
        "last_seen": datetime.now(),
        "last_greeting": datetime(1950, 1, 1),
        "seen_count": 1,
        "seen_frames": 1,
        "face_image": face_image,
        "encoding": face_encoding,
        "person": known_persons['Unknown']
    })


class Doorcam():
    def __init__(self, known_faces={}, known_persons={}):
        self.video_capture = cv2.VideoCapture(0)
        self.scale = 4
        self.known_faces = known_faces
        self.known_persons = known_persons
        self.matches = None

        self.keywords = ['picovoice', 'computer', 'terminator']
        self.pv_handle = pvporcupine.create(keywords=self.keywords)

        self.r = sr.Recognizer()

        self.py_audio = pyaudio.PyAudio()

        self.audio_stream = self.py_audio.open(
            rate=self.pv_handle.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            #frames_per_buffer=handle.frame_length,
            #input_device_index=17
        )

        self.run_audio = True
        self.thread = Thread(target=self.thread_function)
        self.thread.start()

    
    def recognize(self):
        self.ret, self.frame = self.video_capture.read()
        if self.ret:
            small_frame = cv2.resize(self.frame, (0, 0), fx=1/self.scale, fy=1/self.scale)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the face locations and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        

            if face_encodings != []:
                print("Person Detected")
            # Loop through each detected face and see if it is one we have seen before
            # If so, we'll give it a label that we'll draw on top of the video.
            face_labels = []
            self.matches = []
            for face_location, face_encoding in zip(face_locations, face_encodings):
                # See if this face is in our list of known faces.
                match, metadata = lookup_known_face(self.known_faces, face_encoding)
                self.matches.append(match)
                print(metadata['person'] if metadata != None else "Unrecognized Person")
                
                # If we found the face, label the face with some useful information.
                if metadata is not None:
                    #print(str(metadata['person']) + "Detected")
                    time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                    name = metadata['person']['name']
                    face_label = f"{name}. At door {int(time_at_door.total_seconds())}s"
                    if (datetime.now() - metadata['last_greeting']).total_seconds() > 60*5:
                        #p.play(metadata['person']['greeting'])
                        metadata['last_greeting'] = datetime.now()
                else:
                    print("Unrecognized Person")

                    top, right, bottom, left = face_location
                    face_image = small_frame[top:bottom, left:right]
                    face_image = cv2.resize(face_image, (150, 150))

                    register_new_face(self.known_faces, self.known_persons, face_encoding, face_image)
                    save_known(known_faces, known_persons)
            cv2.imshow('Video', self.frame)
    
    def audio_frame(self):
        pcm = self.audio_stream.read(self.pv_handle.frame_length)
        audio_frame = struct.unpack_from("h" * self.pv_handle.frame_length, pcm)
        keyword_index = self.pv_handle.process(audio_frame)
        a = np.max(audio_frame) 
        #print(a, end=" ")
        if a == 0:
            print('Audio frame is 0. Please check microphone settings.')
        if keyword_index >= 0:
            # detection event logic/callback
            
            print(f'Keyword recognized: {keyword_index}')
            if len(self.matches) == 0:
                wave_obj = sa.WaveObject.from_wave_file("greetings/no_person_in_frame.wav")
                play_obj = wave_obj.play()
                play_obj.wait_done()

            else:
                match = self.matches[-1]
                print('New greeting' + str(match))
                wave_obj = sa.WaveObject.from_wave_file("greetings/please_enter_greeting.wav")
                play_obj = wave_obj.play()
                play_obj.wait_done()

                print("Recording greeting")

                for i in range(150): # Throw out the first 100
                    pcm = self.audio_stream.read(self.pv_handle.frame_length)

                audio_record = []
                for i in range(200):
                    pcm = self.audio_stream.read(self.pv_handle.frame_length)
                    audio_record.append(list(struct.unpack_from("h" * self.pv_handle.frame_length, pcm)))
                wavdata = np.array(audio_record).flatten()
                wavdata = (wavdata/np.max(wavdata)*30000).astype(np.int16)
                print("Audio Length: " + str(len(wavdata)))
                print(np.max(wavdata))
                filename = "greetings/greeting" + str(match) + ".wav"
                wavfile.write(filename, self.pv_handle.sample_rate, wavdata)
                print("Playing greeting" + str(match) + ".wav")
                wave_obj = sa.WaveObject.from_wave_file(filename)
                play_obj = wave_obj.play()
                play_obj.wait_done()
                print("Audio Recorded")

                known_faces[match]["person"] = {'name': 'Person ' + str(match), 'greeting': filename}

    def thread_function(self):
        while self.run_audio == True:
            self.audio_frame()
    
    def end(self):
        self.run_audio = False
        self.pv_handle.delete()
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.py_audio.terminate()

        

if __name__ == "__main__":

    known_faces, known_persons = load_known()

    door = Doorcam(known_faces, known_persons)

    while True:
        door.recognize()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            door.end()
            break

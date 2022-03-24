"""
Saving and loading utilities

"""

import pickle

filename = "known_faces_and_persons.dat"

def save_known(known_faces, known_persons):
    with open(filename, "wb") as face_data_file:
        face_data = [known_faces, known_persons]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")


def load_known():
    try:
        with open(filename, "rb") as face_data_file:
            known_faces, known_persons = pickle.load(face_data_file)
            print("Known faces and persons loaded from disk.")

    except:
        print("No previous face data found - starting with a blank known face list.")
        known_faces = []
        known_persons = {}
        known_persons['Unknown'] = {'name': 'Unknown',
                                    'greeting': 'greetings/recording1.wav'}
        
    return known_faces, known_persons

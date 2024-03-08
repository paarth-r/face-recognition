from PIL import Image, ImageDraw
import face_recognition
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import face_recognition
import os, shutil
from matplotlib.patches import PathPatch
import cv2


def recognise(url1, url1name, url2, url2name, url3, savename):

  face1_image = face_recognition.load_image_file(url1)
  face1_encoding = face_recognition.face_encodings(face1_image)[0]

  face2_image = face_recognition.load_image_file(url2)
  face2_encoding = face_recognition.face_encodings(face2_image)[0]

  known_face_encodings = [
      face1_encoding,
      face2_encoding
  ]
  known_face_names = [
      url1name,
      url2name
  ]

  # Load an image with an unknown face
  unknown_image = face_recognition.load_image_file(url3)

  # Find all the faces and face encodings in the unknown image
  face_locations = face_recognition.face_locations(unknown_image)
  face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

  # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
  # See http://pillow.readthedocs.io/ for more about PIL/Pillow
  pil_image = Image.fromarray(unknown_image)
  # Create a Pillow ImageDraw Draw instance to draw with
  draw = ImageDraw.Draw(pil_image)

  # Loop through each face found in the unknown image
  for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      
      # See if the face is a match for the known face(s)
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

      name = "Unknown"

      # If a match was found in known_face_encodings, just use the first one.
      try:
          first_match_index = matches.index(True)
          name = known_face_names[first_match_index]
      except:
          name = "Unknown"

      # Draw a box around the face using the Pillow module
      draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

      # Draw a label with a name below the face
      text_width, text_height = draw.textsize(name)
      draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
      draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


  # Remove the drawing library from memory as per the Pillow docs
  del draw

  # Display the resulting image
  plt.imshow(pil_image, aspect='auto')

  # Save the resulting image
  pil_image.save(f"/Users/paarth/Desktop/projekts/facerecognition/Face-ID/oneshotfacerecog/recognised/{savename}.jpg")

imageknown = "/Users/paarth/Desktop/projekts/facerecognition/Face-ID/oneshotfacerecog/known/paarthface.jpg"
face1name = "Paarth"
imageknown2 = "/Users/paarth/Desktop/projekts/facerecognition/Face-ID/oneshotfacerecog/known/mama.jpg"
face2name = "Ashish"
detectimage = "/Users/paarth/Desktop/projekts/facerecognition/Face-ID/oneshotfacerecog/data/frame_6.jpg"
images = [img for img in os.listdir("/Users/paarth/Desktop/projekts/facerecognition/Face-ID/oneshotfacerecog/unknown") if img.endswith(".png")]
for n, imagename in enumerate(images):
    recognise(imageknown, face1name, imageknown2, face2name, '/Users/paarth/Desktop/projekts/facerecognition/Face-ID/oneshotfacerecog/recognised/' + str(imagename), f"identified_frame_{n}")




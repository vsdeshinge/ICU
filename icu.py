import cv2
import numpy as np
import face_recognition
import os
import datetime


def find_encodings(images):

    encode_list = []
    for img in images:
        # Convert to RGB for face_recognition
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


# Prepare known images and encodings
path = 'images'
images = []
classNames = []
for c1 in os.listdir(path):
    cur_img = cv2.imread(f"{path}/{c1}")
    if cur_img is not None:  # Handle potential errors during image loading
        images.append(cur_img)
        classNames.append(os.path.splitext(c1)[0])

if not images:
    print("Error: No images found in the 'images' directory.")
    exit()  # Terminate if no known faces are available

encode_list_known = find_encodings(images)
print('Encoding Complete')

# Start webcam capture
cap = cv2.VideoCapture(0)


# Start time
start_time = time.time()


now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
print("Initial timestamp:", timestamp)


name_set = set()

total_list = []

# timer for to add set to list
set_timer = time.time()

while True:
    success, img = cap.read()

    if not success:
        print("Error: Failed to read from webcam.")
        break

    # Resize and convert to RGB for face recognition
    img_s = cv2.resize(img, (0, 0), None, .25, .25)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame
    faces_cur_frame = face_recognition.face_locations(img_s)
    encodes_cur_frame = face_recognition.face_encodings(img_s, faces_cur_frame)

    # Match detected faces with known faces
    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(
            encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(
            encode_list_known, encode_face)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = classNames[match_index].upper()
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            name_set.add(name)
            # print(name, timestamp)
            y1, x2, y2, x1 = face_loc
            cv2.rectangle(img, (x1*4, y1*4), (x2*4, y2*4), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2), (x2, y2), (0, 255, 0), cv2.FILLED)

            cv2.putText(img, f"{timestamp} - {name}", (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Webcam', img)

    # add as list
    if time.time() - set_timer >= 5:
        set_timer = time.time()
        total_list.append(list(name_set))
        name_set.clear()

     # Check if 10 seconds have elapsed
    if time.time() - start_time > 20:
        print("Successful run for 10 seconds.")
        break

    # Handle key press for quitting
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


print(total_list)
# Release resources
cap.release()
cv2.destroyAllWindows()


lattened_list = [element for sublist in total_list for element in sublist]

flattendance_list = np.array(lattened_list)

# converting list to upper
name_list = np.char.upper(name_list)
name_list

name_count_dict = {}


for name in flattendance_list:
    if name in name_count_dict:
        name_count_dict[name] += 1
    else:
        name_count_dict[name] = 1

print(name_count_dict)


threshold = 2

# Create a list of names where the count is greater than the threshold
names_with_greater_count = [
    name for name, count in name_count_dict.items() if count > threshold]

print("List of names with count greater than", threshold, ":")
print(names_with_greater_count)

mask = np.logical_not(np.array(
    [any(name in item for name in names_with_greater_count) for item in name_list]))

# Apply the mask to filter out elements from the total_list
filtered_list = name_list[mask]

print("Filtered list:")
print(filtered_list)

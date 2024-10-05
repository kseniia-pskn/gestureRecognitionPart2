import cv2
import os
import tensorflow as tf
import frameextractor as fe
import handshape_feature_extractor as hfe
import csv
import re as regex


class GestureDetail:
    def __init__(self, gesture_key, gesture_name, output_label):
        self.gesture_key = gesture_key
        self.gesture_name = gesture_name
        self.output_label = output_label


class GestureFeature:
    def __init__(self, gesture_detail: GestureDetail, extracted_feature):
        self.gesture_detail = gesture_detail
        self.extracted_feature = extracted_feature


def extract_feature(location, input_file, mid_frame_counter):
    middle_image = cv2.imread(fe.frameExtractor(location + input_file, location + "frames/", mid_frame_counter),
                              cv2.IMREAD_GRAYSCALE)
    response = hfe.HandShapeFeatureExtractor.extract_feature(hfe.HandShapeFeatureExtractor.get_instance(),
                                                             middle_image)
    return response


def decide_gesture_by_file_name(gesture_file_name):
    # Extract the gesture key from the file name
    match = regex.match(r"([A-Za-z0-9]+)_PRACTICE_\d+", gesture_file_name)
    if match:
        gesture_key = match.group(1)
        for x in gesture_details:
            if x.gesture_key.lower() == gesture_key.lower():
                return x
    return None


def determine_gesture(gesture_location, gesture_file_name, mid_frame_counter):
    video_feature = extract_feature(gesture_location, gesture_file_name, mid_frame_counter)

    if video_feature is None:
        print(f"Warning: Could not extract feature for {gesture_file_name}. Skipping.")
        return None

    cos_sin = 1
    recognized_gesture_detail = None

    # Calculate cosine similarity to determine the most similar gesture
    for featureVector in featureVectorList:
        calc_cos_sin = tf.keras.losses.cosine_similarity(
            video_feature,
            featureVector.extracted_feature,
            axis=-1
        )
        if calc_cos_sin < cos_sin:
            cos_sin = calc_cos_sin
            recognized_gesture_detail = featureVector.gesture_detail

    # Log the result or mark it as unrecognized
    if recognized_gesture_detail:
        print(f"{gesture_file_name} calculated gesture {recognized_gesture_detail.gesture_name}")
    else:
        print(f"{gesture_file_name} could not be recognized.")
    
    return recognized_gesture_detail


gesture_details = [
    GestureDetail("Num0", "0", "0"), GestureDetail("Num1", "1", "1"),
    GestureDetail("Num2", "2", "2"), GestureDetail("Num3", "3", "3"),
    GestureDetail("Num4", "4", "4"), GestureDetail("Num5", "5", "5"),
    GestureDetail("Num6", "6", "6"), GestureDetail("Num7", "7", "7"),
    GestureDetail("Num8", "8", "8"), GestureDetail("Num9", "9", "9"),
    GestureDetail("FanDown", "Decrease Fan Speed", "10"),
    GestureDetail("FanOn", "FanOn", "11"), GestureDetail("FanOff", "FanOff", "12"),
    GestureDetail("FanUp", "Increase Fan Speed", "13"),
    GestureDetail("LightOff", "LightOff", "14"), GestureDetail("LightOn", "LightOn", "15"),
    GestureDetail("SetThermo", "SetThermo", "16")
]

# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
featureVectorList = []
path_to_train_data = "traindata/"
count = 0
for file in os.listdir(path_to_train_data):
    if not file.startswith('.') and not file.startswith('frames') and not file.startswith('results'):
        gesture_detail = decide_gesture_by_file_name(file)
        if gesture_detail:
            featureVectorList.append(GestureFeature(gesture_detail, extract_feature(path_to_train_data, file, count)))
        count += 1

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
video_locations = ["test/"]
test_count = 0

with open('results.csv', 'w', newline='') as results_file:
    fieldnames = ['Gesture_Video_File_Name', 'Gesture_Name', 'Output_Label']
    train_data_writer = csv.DictWriter(results_file, fieldnames=fieldnames)
    train_data_writer.writeheader()

    for video_location in video_locations:
        for test_file in os.listdir(video_location):
            if not test_file.startswith('.') and not test_file.startswith('frames') \
                    and not test_file.startswith('results'):
                
                recognized_gesture_detail = determine_gesture(video_location, test_file, test_count)
                test_count += 1

                # Write result to CSV
                if recognized_gesture_detail:
                    train_data_writer.writerow({
                        'Gesture_Video_File_Name': test_file,
                        'Gesture_Name': recognized_gesture_detail.gesture_name,
                        'Output_Label': recognized_gesture_detail.output_label
                    })
                else:
                    train_data_writer.writerow({
                        'Gesture_Video_File_Name': test_file,
                        'Gesture_Name': "Unrecognized",
                        'Output_Label': "N/A"
                    })

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from contextlib import redirect_stdout
from sklearn import svm
import face_recognition
import os
import cv2
import time


print("[INFO] initilizing paths")

start_time=time.time()

program_dir = os.getcwd()
input_dir = os.path.join(program_dir,'input')
output_dir = os.path.join(program_dir,'output')
image_dir = os.path.join(input_dir,'image')
video_dir = os.path.join(input_dir,'video')
random_dir = os.path.join(input_dir,'random')

image_name = os.listdir(image_dir)[0]
video_name = os.listdir(video_dir)[0]

image_path = os.path.join(image_dir, image_name)
video_path = os.path.join(video_dir, video_name)

start_time2=time.time()

print("[INFO] known image encoding")

known_encodings = []
known_names = []
for image_name in os.listdir(image_dir) :
    image_path = os.path.join(image_dir, image_name)
    known_image = face_recognition.load_image_file(image_path)
    known_encoding = face_recognition.face_encodings(known_image)[0]
    known_encodings.append(known_encoding)
    known_names.append('known')

for image_name in os.listdir(random_dir) :
    image_path = os.path.join(random_dir, image_name)
    known_image = face_recognition.load_image_file(image_path)
    known_encoding = face_recognition.face_encodings(known_image)[0]
    known_encodings.append(known_encoding)
    known_names.append('unknown')

clf = svm.SVC(gamma ='scale')
clf.fit(known_encodings, known_names)

start_time3=time.time()

video = cv2.VideoCapture(video_path)
video_fps = video.get(cv2.CAP_PROP_FPS)
video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video_length = video_frame_count/video_fps

matches, match_count = list(), 0
timer, my_fps, my_skip = 0.0, 1/2, 5/2
os.chdir(output_dir)

print('[INFO] finding matches')
while (timer <= video_length) :
    #print('setting video')
    video.set(cv2.CAP_PROP_POS_MSEC,timer*1000)
    #print('reading image')
    hasFrames, image = video.read()
    # print('getting encodings')
    encodings = face_recognition.face_encodings(image)
    # print('finding current time')
    minutes = int(timer//60)
    seconds = int(timer%60)
    # print('no encodings check')
    if (encodings == []) :
        timer = timer + my_fps
        continue
    # print('predicting')
    if 'known' in clf.predict(encodings) :
        match_count = match_count + 1
        matches.append(timer)
        timer = timer + my_skip
        print("[INFO] found match no {} at {}:{}".format(match_count,minutes, seconds))
    else :
        timer = timer + my_fps
    print("[INFO] finished {}:{}".format(minutes, seconds))

start_time4 = time.time()

print("[INFO] found {0} matches".format(match_count))

n = len(matches)
clip_no = 1
i = 1

while(i < n) :
    clip_start = max(matches[i]-2.5,0)
    while((i+1 < n) and (matches[i+1] - matches[i] <= 6)) :
        i = i+1
    clip_stop = min(matches[i]+2.5,matches[n-1])
    clip_name = 'clip'+str(clip_no)+'.mp4'
    with open('log.txt', 'w') as f:
        with redirect_stdout(f):
            ffmpeg_extract_subclip(video_path, clip_start, clip_stop, targetname=clip_name)
    clip_no = clip_no+1
    i = i+1
print("[INFO] successfully completed in : {}".format(time.strftime('%H:%M:%S', time.gmtime(start_time4-start_time))))

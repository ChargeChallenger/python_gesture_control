from keras.models import Sequential, load_model, model_from_json
import cv2
import numpy as np
import pyautogui

# Load neuro model
hand_model = load_model('hand_model_gray.hdf5', compile=False)

classes = {
    0: 'fist',
    1: 'five',
    2: 'point',
    3: 'swing'
}

# Counter for delay between actions
counter = 0
# Switcher for gesture controls
isGestureContolOn = False

# Function for gesture control
def control(predict, count):
    if predict == 0:
        if count <= 0:
            pyautogui.hotkey('volumedown')
            count = 10
        else:
            count -= 1
    elif predict == 1:
        if count <= 0:
            pyautogui.hotkey('volumemute')
            count = 10
        else:
            count -= 1
    elif predict == 2:
        if counter <= 0:
            pyautogui.hotkey('volumeup')
            count = 10
        else:
            count -= 1
    elif predict == 3:
        if count <= 0:
            pyautogui.hotkey('playpause')
            count = 10
        else:
            count -= 1
    return count

# Function for finging differences between images
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err
    
# Helper function for applying a mask to an array
def mask_array(array, imask):
    if array.shape[:2] != imask.shape:
        raise Exception("Shapes of input and imask are incompatible")
    output = np.zeros_like(array, dtype=np.uint8)
    for i, row in enumerate(imask):
        output[i, row] = array[i, row]
    return output


# Begin capturing video
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Could not open video")
    sys.exit()


# Read first frame
ok, frame = video.read()
if not ok:
    print("Cannot read video")
    sys.exit()
# Use the first frame as an initial background frame
bg = frame.copy()


# Kernel for erosion and dilation of masks
kernel = np.ones((3,3),np.uint8)


# Display positions (pixel coordinates)
positions = {
    'hand_pose': (15, 40), # hand pose text
    'fps': (15, 20), # fps counter
}

bbox_initial = (116, 116, 170, 170) # Starting position for bounding box
bbox = bbox_initial
hand_crop_temp = bg[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
hand_crop_bg = np.zeros((hand_crop_temp.shape[0], hand_crop_temp.shape[1])) # Hand bg for checking differences 

# Capture, process, display loop    
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break
        
    # Start timer
    timer = cv2.getTickCount()

    # Processing
    # First find the absolute difference between the two images
    diff = cv2.absdiff(bg, frame)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Threshold the mask
    th, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    # Opening, closing and dilation
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img_dilation = cv2.dilate(closing, kernel, iterations=2)
    # Get mask indexes
    imask = img_dilation > 0
    # Get foreground from mask
    foreground = mask_array(frame, imask)
    foreground_display = foreground.copy()    
        
    # Use numpy array indexing to crop the foreground frame
    hand_crop = img_dilation[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    # Cheking differences in hand crop
    if mse(hand_crop, hand_crop_bg) < 13000:
        cv2.putText(foreground_display, "hand pose: none", positions['hand_pose'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    else:
        try:
            # Resize cropped hand and make prediction on gesture
            hand_crop_resized = np.expand_dims(cv2.resize(hand_crop, (54, 54)), axis=0).reshape((1, 54, 54, 1))
            prediction = hand_model.predict(hand_crop_resized)
            predi = prediction[0].argmax() # Get the index of the greatest confidence
            gesture = classes[predi]
            if isGestureContolOn == True:
                counter = control(predi, counter)
        
            for i, pred in enumerate(prediction[0]):
                # Draw confidence bar for each gesture
                barx = positions['hand_pose'][0]
                bary = 60 + i*60
                bar_height = 20
                bar_length = int(400 * pred) + barx # calculate length of confidence bar
            
                # Make the most confidence prediction green
                if i == predi:
                    colour = (0, 255, 0)
                else:
                    colour = (0, 0, 255)
        
            cv2.putText(foreground_display, "hand pose: {}".format(gesture), positions['hand_pose'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        except Exception as ex:
            cv2.putText(foreground_display, "hand pose: error", positions['hand_pose'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
        
    # Draw bounding box
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(foreground_display, p1, p2, (255, 0, 0), 2, 1)
        
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # Display FPS on frame
    cv2.putText(foreground_display, "FPS : " + str(int(fps)), positions['fps'], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 170, 50), 2)

    # Display diff
    cv2.imshow("diff", diff)
    # Display thresh
    cv2.imshow("thresh", thresh)
    # Display mask
    cv2.imshow("img_dilation", img_dilation)
    try:
        # Display hand_crop
        cv2.imshow("hand_crop", hand_crop)
    except:
        pass
    # Display foreground_display
    cv2.imshow("foreground_display", foreground_display)
    
    
    k = cv2.waitKey(1) & 0xff
    
    if k == 27: break # ESC pressed
    elif k == 114 or k == 108: 
        # r pressed
        bg = frame.copy()
        bbox = bbox_initial
    elif k == 32:
        # space pressed
        isGestureContolOn = not(isGestureContolOn)
        
cv2.destroyAllWindows()
video.release()
